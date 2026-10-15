from __future__ import annotations

from typing import Any, Tuple

import inspect
import os
import sys
import torch
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer

from emabench.config import RunConfig


def _get_instruct_response_part(tokenizer) -> Tuple[str, str]:
    prefix_conversation = [
        {"role": "user", "content": "ignore"},
        {"role": "assistant", "content": "ignore"},
    ]
    example_conversation = prefix_conversation + [
        {"role": "user", "content": "<user message content>"}
    ]
    example_text = tokenizer.apply_chat_template(
        example_conversation, add_generation_prompt=False, tokenize=False
    )
    options = [
        # Qwen2.5 / Qwen2 (the default model here)
        ("<|im_start|>user\n", "<|im_start|>assistant\n"),
        # Llama 3.x
        (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        (
            "<|start_header_id|>user<|end_header_id|>\n",
            "<|start_header_id|>assistant<|end_header_id|>\n",
        ),
        # Mistral
        ("[INST]", "[/INST]"),
        # DeepSeek
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for instruction_part, response_part in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, "")
    instruction_part, _ = main_part.split("<user message content>")
    response_part = tokenizer.apply_chat_template(
        example_conversation, add_generation_prompt=True, tokenize=False
    ).replace(example_text, "")
    return instruction_part, response_part


class DefenseSFTTrainer(SFTTrainer):
    def __init__(self, *args, defense=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.defense = defense

    def _split_model_inputs(self, model, inputs: dict) -> tuple[dict, dict]:
        sig = inspect.signature(model.forward)
        accepted = set(sig.parameters.keys())
        model_inputs = {}
        aux_inputs = {}
        for k, v in inputs.items():
            if k in accepted or k in {"labels", "return_dict"}:
                model_inputs[k] = v
            else:
                aux_inputs[k] = v
        return model_inputs, aux_inputs

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        model_inputs, aux_inputs = self._split_model_inputs(model, inputs)
        outputs = model(**model_inputs, return_dict=True)
        logits = outputs.logits

        labels = model_inputs.get("labels")
        if labels is None:
            return super().compute_loss(
                model,
                model_inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        base_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        if self.defense is not None:
            full_batch = dict(model_inputs)
            full_batch.update(aux_inputs)
            base_loss = self.defense.modify_loss(base_loss, logits, full_batch, model)

        return (base_loss, outputs) if return_outputs else base_loss


def _stabilize_training_args_class_for_pickle(args: Any) -> None:
    """Avoid PicklingError when TRL/Unsloth reloads SFTConfig at runtime.

    In some Unsloth + TRL setups, ``args`` is an instance of a previous
    ``trl.trainer.sft_config.SFTConfig`` class object, while the module global
    ``SFTConfig`` points to a reloaded class object. ``pickle`` then raises:

        "it's not the same object as trl.trainer.sft_config.SFTConfig"

    Rebinding the module attribute to the *actual* class used by ``args`` makes
    the class identity check pass during checkpoint saves.
    """
    cls = args.__class__
    module = sys.modules.get(cls.__module__)
    if module is None:
        return
    name = cls.__name__
    if getattr(module, name, None) is not cls:
        setattr(module, name, cls)


class _PreserveBatchFieldsCollator:
    def __init__(self, base_collator, field_names: list[str]):
        self.base_collator = base_collator
        self.field_names = field_names

    def __call__(self, features):
        preserved: dict[str, list] = {name: [] for name in self.field_names}
        stripped = []
        for feat in features:
            feat = dict(feat)
            for name in self.field_names:
                preserved[name].append(feat.pop(name, 0))
            stripped.append(feat)

        batch = self.base_collator(stripped)
        for name in self.field_names:
            batch[name] = torch.tensor(preserved[name], dtype=torch.float)
        return batch


def train_sft(
    cfg: RunConfig,
    model: Any,
    tokenizer: Any,
    dataset,
    test_dataset,
    defense=None,
):
    passthrough_keys = [
        k for k in ["is_interleaved", "interleaved", "has_context"] if k in dataset.column_names
    ]

    def apply_chat_template(examples):
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            # add_generation_prompt must be False for training: True would
            # append a dangling empty assistant header (e.g. Qwen's
            # "<|im_start|>assistant\n") AFTER the real response, teaching the
            # model to emit EOS right after opening a turn. The template
            # already closes the assistant turn; only append EOS manually for
            # templates whose turn-end token differs from eos_token.
            text = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=False,
            )
            if tokenizer.eos_token and not text.rstrip().endswith(tokenizer.eos_token):
                text = text + tokenizer.eos_token
            texts.append(text)

        out = {"text": texts}
        for key in passthrough_keys:
            if key in examples:
                out[key] = examples[key]
        return out

    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)

    learning_rate = cfg.train.learning_rate
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)

    args = TrainingArguments(
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        warmup_steps=cfg.train.warmup_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim=cfg.train.optim,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        seed=cfg.train.seed,
        report_to=[],
        num_train_epochs=cfg.train.epochs,
        max_steps=cfg.train.max_steps or -1,
        save_steps=cfg.train.save_steps,
        output_dir=cfg.train.output_dir,
        remove_unused_columns=False,
    )

    trainer = DefenseSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=args,
        defense=defense,
    )

    if cfg.train.train_on_responses_only:
        instruction_part, response_part = _get_instruct_response_part(tokenizer)
        trainer.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )

    preserve_fields = [k for k in ["is_interleaved"] if k in dataset.column_names]
    if preserve_fields:
        trainer.data_collator = _PreserveBatchFieldsCollator(
            trainer.data_collator, preserve_fields
        )

    _stabilize_training_args_class_for_pickle(trainer.args)
    trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    return trainer
