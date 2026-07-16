#!/bin/bash

#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=response-embeddings
#SBATCH --output=response-embeddings-%j.out
#SBATCH --error=response-embeddings-%j.err

set -euo pipefail

module purge
module load python

source "$HOME/.venvs/thesis-embeddings/bin/activate"

cd "$SLURM_SUBMIT_DIR"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${SCRATCH:-$HOME/.cache}/huggingface"

mkdir -p "$HF_HOME"
mkdir -p data/processed

python slurm/generate_embeddings.py \
    --input slurm/response_text.csv \
    --output data/processed/response_embeddings.npy \
    --batch-size 256