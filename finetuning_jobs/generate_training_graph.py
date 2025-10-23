#!/usr/bin/env python3
"""
Script to parse training logs and generate training loss graphs for FSDP and DDP approaches.
"""

import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file_path):
    """Parse training log file and extract loss data."""
    losses = []
    steps = []
    epochs = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Look for lines with loss data in the format: {'loss': 1.4857, ...}
            if "'loss':" in line and "grad_norm" in line:
                try:
                    # Extract the dictionary string
                    match = re.search(r"\{.*\}", line)
                    if match:
                        data_str = match.group(0)
                        # Replace single quotes with double quotes for JSON parsing
                        data_str = data_str.replace("'", '"')
                        data = json.loads(data_str)
                        
                        losses.append(data['loss'])
                        steps.append(len(losses) * 10)  # Assuming logging every 10 steps
                        epochs.append(data['epoch'])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not parse line: {line.strip()}")
                    continue
    
    return steps, losses, epochs

def create_training_loss_graph():
    """Create training loss graph comparing FSDP and DDP approaches."""
    
    # Parse FSDP log
    fsdp_log_path = "fsdp/run/fsdp-llama-232664.log"
    ddp_log_path = "ddp/run/ddp-llama-232982.log"
    
    # Parse both logs
    fsdp_steps, fsdp_losses, fsdp_epochs = parse_training_log(fsdp_log_path)
    ddp_steps, ddp_losses, ddp_epochs = parse_training_log(ddp_log_path)
    
    print(f"FSDP: Found {len(fsdp_losses)} loss points")
    print(f"DDP: Found {len(ddp_losses)} loss points")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot FSDP training loss
    plt.plot(fsdp_steps, fsdp_losses, 'b-', linewidth=2, label='FSDP (No Quantization)', alpha=0.8)
    
    # Plot DDP training loss
    plt.plot(ddp_steps, ddp_losses, 'r-', linewidth=2, label='DDP (4-bit Quantization)', alpha=0.8)
    
    # Add some smoothing for better visualization
    if len(fsdp_losses) > 10:
        fsdp_smooth = np.convolve(fsdp_losses, np.ones(5)/5, mode='valid')
        fsdp_smooth_steps = fsdp_steps[2:-2]  # Adjust steps for smoothed data
        plt.plot(fsdp_smooth_steps, fsdp_smooth, 'b--', linewidth=1, alpha=0.6, label='FSDP (Smoothed)')
    
    if len(ddp_losses) > 10:
        ddp_smooth = np.convolve(ddp_losses, np.ones(5)/5, mode='valid')
        ddp_smooth_steps = ddp_steps[2:-2]  # Adjust steps for smoothed data
        plt.plot(ddp_smooth_steps, ddp_smooth, 'r--', linewidth=1, alpha=0.6, label='DDP (Smoothed)')
    
    # Customize the plot
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Comparison: FSDP vs DDP', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add final loss annotations
    if fsdp_losses:
        plt.annotate(f'FSDP Final: {fsdp_losses[-1]:.3f}', 
                    xy=(fsdp_steps[-1], fsdp_losses[-1]), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                    fontsize=10)
    
    if ddp_losses:
        plt.annotate(f'DDP Final: {ddp_losses[-1]:.3f}', 
                    xy=(ddp_steps[-1], ddp_losses[-1]), 
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                    fontsize=10)
    
    # Set axis limits for better visualization
    plt.xlim(0, max(max(fsdp_steps) if fsdp_steps else 0, max(ddp_steps) if ddp_steps else 0))
    all_losses = fsdp_losses + ddp_losses
    if all_losses:
        plt.ylim(min(all_losses) * 0.9, max(all_losses) * 1.1)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "report/build/training_loss_comparison.png"
    Path("report/build").mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training loss graph saved to: {output_path}")
    
    # Also save as PDF for LaTeX
    pdf_path = "report/build/training_loss_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Training loss graph (PDF) saved to: {pdf_path}")
    
    plt.show()
    
    return output_path, pdf_path

def print_summary_stats(fsdp_steps, fsdp_losses, ddp_steps, ddp_losses):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("TRAINING LOSS SUMMARY")
    print("="*60)
    
    if fsdp_losses:
        print(f"FSDP (Fully Sharded Data Parallel):")
        print(f"  - Total steps: {len(fsdp_losses)}")
        print(f"  - Initial loss: {fsdp_losses[0]:.4f}")
        print(f"  - Final loss: {fsdp_losses[-1]:.4f}")
        print(f"  - Loss reduction: {fsdp_losses[0] - fsdp_losses[-1]:.4f}")
        print(f"  - Min loss: {min(fsdp_losses):.4f}")
        print(f"  - Max loss: {max(fsdp_losses):.4f}")
    
    print()
    
    if ddp_losses:
        print(f"DDP (Distributed Data Parallel):")
        print(f"  - Total steps: {len(ddp_losses)}")
        print(f"  - Initial loss: {ddp_losses[0]:.4f}")
        print(f"  - Final loss: {ddp_losses[-1]:.4f}")
        print(f"  - Loss reduction: {ddp_losses[0] - ddp_losses[-1]:.4f}")
        print(f"  - Min loss: {min(ddp_losses):.4f}")
        print(f"  - Max loss: {max(ddp_losses):.4f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("Generating training loss comparison graph...")
    
    # Parse logs
    fsdp_log_path = "fsdp/run/fsdp-llama-232664.log"
    ddp_log_path = "ddp/run/ddp-llama-232982.log"
    
    fsdp_steps, fsdp_losses, fsdp_epochs = parse_training_log(fsdp_log_path)
    ddp_steps, ddp_losses, ddp_epochs = parse_training_log(ddp_log_path)
    
    # Print summary statistics
    print_summary_stats(fsdp_steps, fsdp_losses, ddp_steps, ddp_losses)
    
    # Create and save the graph
    png_path, pdf_path = create_training_loss_graph()
    
    print(f"\nGraphs generated successfully!")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
