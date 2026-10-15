#!/bin/bash
# One emabench config on one GPU; the three seeds run as a job array (one seed per task).
# Fill in the <PLACEHOLDERS> for your cluster, then:
#
#   EMA_CONFIG=configs/security/interleaving_wglift.yaml sbatch slurm/train_eval.slurm
#   emabench --config configs/security/interleaving_wglift.yaml      # afterwards, on any node: pools the seeds -> summary.json
#
# A seed resumes where it stopped (adapter -> skip training, checkpoint -> resume,
# generations.jsonl -> resume sampling, result.json -> skip the eval), so a timed-out
# array task can simply be resubmitted. 35-50 minutes per seed on one H100.
#
#SBATCH --job-name=emabench
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=<LOG_DIR>/%A_%a_%x.out
#SBATCH --error=<LOG_DIR>/%A_%a_%x.out
set -euo pipefail

# OPENAI_API_KEY must be in the job's environment (the em_judge evaluations call the judge);
# e.g. export it in a file sourced here, never write it into this script or a config.
# source ~/.secrets

cd "$SLURM_SUBMIT_DIR"                 # the repository root
source .venv/bin/activate              # created with `uv sync` (see README)

export HF_HOME="<HF_CACHE_DIR>"        # model + tinyBenchmarks datasets, pre-downloaded if compute nodes are offline
# export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

CONFIG="${EMA_CONFIG:?set EMA_CONFIG=configs/<dataset>/<arm>.yaml}"
export EMA_SEEDS="${EMA_SEEDS:-${SLURM_ARRAY_TASK_ID:-0}}"   # this array task owns one seed
: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"

echo "host=$(hostname) job=${SLURM_JOB_ID} config=${CONFIG} seed=${EMA_SEEDS} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
emabench --config "$CONFIG"
echo "DONE seed=${EMA_SEEDS} end=$(date -Is)"
