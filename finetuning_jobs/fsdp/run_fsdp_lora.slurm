#!/bin/bash
#SBATCH --job-name=fsdp-llama
#SBATCH --output=fsdp-llama-%j.log
#SBATCH --error=fsdp-llama-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --gpus=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1

# Set environment variables for distributed training
export NCCL_DEBUG=INFO
export TORCH_FR_BUFFER_SIZE=1048576
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

# Hugging Face authentication (replace with your actual token)
export HF_TOKEN="placeholder"

# Ensure environment variables are passed to child processes
export -p

# Activate virtual environment
source ~/.bashrc

if [ -d "fsdp-env" ]; then
    echo "Activating existing virtual environment..."
    source fsdp-env/bin/activate
else
    echo "Virtual environment not found. Creating..."
    python3 -m venv fsdp-env
    source fsdp-env/bin/activate
fi

echo "Installing required packages..."
pip install --upgrade pip

# Install PyTorch with CUDA support (exact version from notebook)
pip install torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# Install ML packages with exact versions
pip install transformers==4.57.1
pip install peft==0.17.1
pip install bitsandbytes==0.48.1
pip install accelerate==1.10.1
pip install trl==0.24.0
pip install datasets==4.2.0


echo "Package installation completed!"

mkdir -p fsdp_results


srun python train_fsdp_lora.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name databricks/databricks-dolly-15k \
    --output_dir fsdp_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_length 1024 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --save_steps 100 \
    --logging_steps 10 \
    --warmup_ratio 0.03 \
    --fp16 \
    --dataloader_pin_memory \
    --remove_unused_columns

echo "Training completed successfully!"
