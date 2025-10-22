#!/usr/bin/env python3
"""
FSDP LoRA Fine-tuning Script for LLaMA-2-7B on Dolly-15K
Designed for NUS SoC SLURM cluster with multiple GPUs
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import Trainer
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
from functools import partial
import json
import platform
import psutil
from datetime import datetime
from transformers import TrainerCallback

class GPUMemoryCallback(TrainerCallback):
    """Callback to log GPU memory usage during training"""
    
    def __init__(self, rank, logging_steps=10):
        self.rank = rank
        self.logging_steps = logging_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.logging_steps == 0:
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
        
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['MASTER_ADDR'] = os.environ.get('SLURM_LAUNCH_NODE_IPADDR', 'localhost')
        os.environ['MASTER_PORT'] = '12355'
        if rank == 0:
            print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
            print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        
        print(f"Running on SLURM cluster with rank: {rank}, world_size: {world_size}, local_rank: {local_rank}")
    else:
        # Local testing
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        print(f"WARNING: Running locally with rank: {rank}, world_size: {world_size}, local_rank: {local_rank}")
        os.exit(1)

    
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=local_rank
    )
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_distributed():
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
    """Get system and hardware information"""
    if rank == 0:
        print("=" * 60)
        print("SYSTEM INFORMATION")
        print("=" * 60)
        print(f"Platform: {platform.platform()}")
        print(f"Python Version: {platform.python_version()}")
        print(f"CPU Count: {psutil.cpu_count()}")
        print(f"Total RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        
        # SLURM environment variables
        if 'SLURM_PROCID' in os.environ:
            print(f"\nSLURM Environment:")
            print(f"  Job ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
            print(f"  Node: {os.environ.get('SLURM_NODEID', 'N/A')}")
            print(f"  Tasks: {os.environ.get('SLURM_NTASKS', 'N/A')}")
            print(f"  CPUs per Task: {os.environ.get('SLURM_CPUS_PER_TASK', 'N/A')}")
            print(f"  Memory: {os.environ.get('SLURM_MEM_PER_NODE', 'N/A')} MB")

def get_gpu_info(rank, local_rank):
    """Get GPU specifications and information for current process only"""
    if rank == 0:
        print("\n" + "=" * 60)
        print("GPU INFORMATION")
        print("=" * 60)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print("=" * 60)
    
    if torch.cuda.is_available():
        # Only log info for this process's GPU
        props = torch.cuda.get_device_properties(local_rank)
        print(f"\n[Rank {rank}] GPU {local_rank}:")
        print(f"  Name: {torch.cuda.get_device_name(local_rank)}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        
        # Current memory usage for this GPU
        allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
        cached = torch.cuda.memory_reserved(local_rank) / 1024**3
        print(f"  Current Memory Allocated: {allocated:.2f} GB")
        print(f"  Current Memory Cached: {cached:.2f} GB")
    else:
        print(f"[Rank {rank}] CUDA not available")



def run_training(args, rank, world_size):
      # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    
    dataset = load_dataset(args.dataset_name)
    full_dataset = dataset['train']

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size) 
    val_size = int(0.1 * total_size)
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    
    train_dataset = full_dataset.select(train_indices).map(format_instruction)
    eval_dataset = full_dataset.select(val_indices).map(format_instruction)

    if rank == 0:
        print(f"Loaded dataset: {args.dataset_name}")
        print(f"Total examples: {total_size}")
        print(f"Train dataset size: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
        print(f"Eval dataset size: {len(eval_dataset)} ({len(eval_dataset)/total_size*100:.1f}%)")
        print(f"Dataset columns: {train_dataset.column_names}")
        print(f"Dataset sample: {train_dataset[0]}")
    

    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    

    model = create_no_fsdp_model(args.model_name, lora_config, rank)
    
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
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
            "fsdp_backward_prefetch": "BACKWARD_PRE",
            "fsdp_forward_prefetch": True,
            "fsdp_use_orig_params": True,
        },
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[GPUMemoryCallback(rank, args.logging_steps)],
    )
    
    if rank == 0:
        print("Starting training...")
        print(f"Total steps: {len(train_dataset) // (args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps) * args.num_train_epochs}")
    
    # Train
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    if rank == 0:
        print(f"Training completed in: {end_time - start_time}")
        print("Saving final model...")
        
        # Save final model
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Save training info
        training_info = {
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "num_gpus": world_size,
            "training_time": str(end_time - start_time),
            "total_steps": trainer.state.global_step,
            "final_loss": trainer.state.log_history[-1].get('train_loss', 'N/A'),
            "lora_config": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout
            }
        }
        
        with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)
        
        print("Training completed successfully!")
        print(f"Results saved to: {args.output_dir}")


def main():
    try:

        # Ensure HF_TOKEN is available
        if 'HF_TOKEN' not in os.environ or os.environ['HF_TOKEN'] == 'placeholder':
            print("Warning: HF_TOKEN not found in environment or is placeholder")
            os.exit(1)
        
        login(new_session=False)
    
        parser = argparse.ArgumentParser(description="FSDP LoRA Fine-tuning")
        parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
        parser.add_argument("--dataset_name", default="databricks/databricks-dolly-15k")
        parser.add_argument("--output_dir", default="fsdp_results")
        parser.add_argument("--num_train_epochs", type=int, default=1)
        parser.add_argument("--per_device_train_batch_size", type=int, default=2)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        parser.add_argument("--learning_rate", type=float, default=2e-4)
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
    
    # Setup distributed training
        rank, world_size, local_rank = setup_distributed()
        get_system_info(rank)
        get_gpu_info(rank, local_rank)
    
        if rank == 0:
            print(f"Starting FSDP training on {world_size} GPUs")
            print(f"Arguments: {args}")
            print(f"Timestamp: {datetime.now()}")
        
        run_training(args, rank, world_size)
    
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
