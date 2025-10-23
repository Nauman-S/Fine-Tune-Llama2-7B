#!/usr/bin/env python3
"""
DDP LoRA Fine-tuning Script for LLaMA-2-7B on Dolly-15K
Designed for NUS SoC SLURM cluster with multiple GPUs
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
import json
import platform
import psutil
from datetime import datetime
from transformers import TrainerCallback

class GPUMemoryCallback(TrainerCallback):
    """Callback to log GPU memory usage during training"""
    
    def __init__(self, rank):
        self.rank = rank
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            allocated = torch.cuda.memory_allocated(self.rank) / 1024**3
            reserved = torch.cuda.memory_reserved(self.rank) / 1024**3
            
            print(f"[Rank {self.rank}] Step {state.global_step}: "
                  f"Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

def setup_distributed():
    """Initialize distributed training for SLURM"""
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        # Set PyTorch distributed environment variables
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Set CUDA device
        torch.cuda.set_device(local_rank)
        
        # Initialize process group
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        print(f"Running on SLURM cluster with rank: {rank}, world_size: {world_size}, local_rank: {local_rank}")
        print(f"Current Memory Allocated: {torch.cuda.memory_allocated(local_rank) / 1024**3:.2f} GB")
        print(f"Current Memory Cached: {torch.cuda.memory_reserved(local_rank) / 1024**3:.2f} GB")
        
        return rank, world_size, local_rank
    else:
        print("Not running on SLURM cluster")
        return 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def format_instruction(example):
    instruction = example['instruction']
    context = example['context']
    response = example['response']
    
    if context:
        prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    
    return {"text": prompt}

def create_no_fsdp_model(model_name, lora_config, rank):
    """Create model with FSDP wrapping"""
    if rank == 0:
        print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=None,
        trust_remote_code=True
    )
    
    model = get_peft_model(model, lora_config)
    model = model.to(dtype=torch.float16)
    
    if rank == 0:
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model

def get_system_info(rank):
    """Get system information"""
    if rank == 0:
        print(f"\n=== System Information ===")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
        print(f"CPU Count: {psutil.cpu_count()}")
        print(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"Timestamp: {datetime.now()}")

def get_gpu_info(rank, local_rank):
    """Get GPU information for current process"""
    if torch.cuda.is_available():
        print(f"\n=== GPU Information (Rank {rank}) ===")
        print(f"GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
        props = torch.cuda.get_device_properties(local_rank)
        print(f"Total Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"Multiprocessors: {props.multi_processor_count}")
        print(f"Compute Capability: {props.major}.{props.minor}")

def format_instruction(example):
    """Format instruction for training"""
    if example.get("input", "").strip():
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['response']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    return {"text": text}

def create_ddp_model(args, rank, local_rank):
    """Create and wrap model with DDP"""
    print(f"[Rank {rank}] Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=None,  # DDP will handle device placement
    )
    
    # Create LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Move model to GPU
    model = model.to(f"cuda:{local_rank}")
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    print(f"[Rank {rank}] Model created and wrapped with DDP")
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="DDP LoRA Fine-tuning")
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset_name", default="databricks/databricks-dolly-15k")
    parser.add_argument("--output_dir", default="ddp_results")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dataloader_pin_memory", action="store_true")
    parser.add_argument("--remove_unused_columns", action="store_true")
    
    args = parser.parse_args()
    
    try:
        # Setup distributed training
        rank, world_size, local_rank = setup_distributed()
        
        # Print system info only on rank 0
        get_system_info(rank)
        get_gpu_info(rank, local_rank)
        
        # Hugging Face authentication
        if 'HF_TOKEN' in os.environ:
            login(token=os.environ['HF_TOKEN'], new_session=False)
        else:
            print("Warning: HF_TOKEN not found in environment")
        
        # Create model and tokenizer
        model, tokenizer = create_ddp_model(args, rank, local_rank)
        
        # Load and prepare dataset
        print(f"[Rank {rank}] Loading dataset...")
        dataset = load_dataset(args.dataset_name)
        
        # Format dataset
        formatted_dataset = dataset.map(format_instruction, remove_columns=dataset["train"].column_names)
        
        # Split dataset (80/10/10)
        train_size = int(0.8 * len(formatted_dataset["train"]))
        val_size = int(0.1 * len(formatted_dataset["train"]))
        
        train_dataset = formatted_dataset["train"].select(range(train_size))
        eval_dataset = formatted_dataset["train"].select(range(train_size, train_size + val_size))
        
        print(f"[Rank {rank}] Dataset size: {len(train_dataset)} train, {len(eval_dataset)} eval")
        print(f"[Rank {rank}] Sample: {train_dataset[0]['text'][:200]}...")
        
        # Training configuration
        training_args = SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            optim="adamw_torch", 
            fp16=args.fp16,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            warmup_ratio=args.warmup_ratio,
            save_total_limit=2,
            report_to="none",
            max_length=args.max_length,
            dataset_text_field="text",
            packing=False,
            do_eval=True,
            eval_steps=50,
            lr_scheduler_type="constant",
            max_grad_norm=0.3,
            group_by_length=True,
            logging_strategy="steps",
            ddp_find_unused_parameters=False,
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            args=training_args,
            callbacks=[GPUMemoryCallback(rank)],
        )
        
        # Start training
        print(f"[Rank {rank}] Starting training...")
        trainer.train()
        
        # Save final model
        if rank == 0:
            print("Saving final model...")
            trainer.save_model()
            tokenizer.save_pretrained(args.output_dir)
        
        print(f"[Rank {rank}] Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()