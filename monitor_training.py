#!/usr/bin/env python3
"""
Simple training monitor for MotionBERT custom training
Alternative to TensorBoard for viewing training progress
"""

import os
import sys
import torch
import json
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def load_checkpoint_info(checkpoint_path):
    """Load basic info from checkpoint file"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        info = {
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'best_acc': checkpoint.get('best_acc', 'Unknown'),
            'lr': checkpoint.get('lr', 'Unknown'),
            'file_size': os.path.getsize(checkpoint_path) / (1024*1024)  # MB
        }
        return info
    except Exception as e:
        return {'error': str(e)}

def parse_log_file(log_path):
    """Parse training logs if available"""
    if not os.path.exists(log_path):
        return None
    
    epochs = []
    train_losses = []
    test_accs = []
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'Epoch' in line and 'Summary:' in line:
                    # Parse epoch summary lines
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'Epoch':
                            epoch = int(parts[i+1])
                            epochs.append(epoch)
                        elif 'Loss:' in part:
                            loss = float(parts[i+1])
                            train_losses.append(loss)
                        elif 'Test' in part and 'Acc@1:' in parts[i+1]:
                            acc = float(parts[i+2].replace('%', ''))
                            test_accs.append(acc)
                            break
        
        return {
            'epochs': epochs,
            'train_losses': train_losses,
            'test_accs': test_accs
        }
    except Exception as e:
        print(f"Error parsing log file: {e}")
        return None

def show_training_status(checkpoint_dir='checkpoint_custom'):
    """Show current training status"""
    print("=" * 60)
    print("🔥 MotionBERT Training Monitor")
    print("=" * 60)
    
    # Check checkpoint files
    latest_chk = os.path.join(checkpoint_dir, 'latest_epoch.bin')
    best_chk = os.path.join(checkpoint_dir, 'best_epoch.bin')
    
    print("\n📁 Checkpoint Status:")
    print("-" * 30)
    
    if os.path.exists(latest_chk):
        info = load_checkpoint_info(latest_chk)
        print(f"✅ Latest Checkpoint: Epoch {info.get('epoch', '?')}")
        print(f"   File size: {info.get('file_size', 0):.1f} MB")
        print(f"   Last modified: {datetime.fromtimestamp(os.path.getmtime(latest_chk))}")
    else:
        print("❌ No latest checkpoint found")
    
    if os.path.exists(best_chk):
        info = load_checkpoint_info(best_chk)
        print(f"🏆 Best Checkpoint: {info.get('best_acc', '?')}% accuracy")
        print(f"   From epoch: {info.get('epoch', '?')}")
        print(f"   File size: {info.get('file_size', 0):.1f} MB")
    else:
        print("❌ No best checkpoint found")
    
    # Check log files
    print("\n📊 Training Logs:")
    print("-" * 30)
    
    log_dir = os.path.join(checkpoint_dir, 'logs')
    if os.path.exists(log_dir):
        log_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
        if log_files:
            print(f"📈 Found {len(log_files)} TensorBoard log files")
            latest_log = max(log_files, key=os.path.getmtime)
            print(f"   Latest: {os.path.basename(latest_log)}")
            print(f"   Size: {os.path.getsize(latest_log)} bytes")
        else:
            print("❌ No TensorBoard logs found")
    else:
        print("❌ No logs directory found")
    
    # Check if training is running
    print("\n🔄 Process Status:")
    print("-" * 30)
    os.system("ps aux | grep -v grep | grep train_action_custom.py | head -3")
    
    print("\n" + "=" * 60)
    
    # Instructions
    print("\n💡 Quick Commands:")
    print("• Start training: python train_action_custom.py")
    print("• View logs: tensorboard --logdir checkpoint_custom/logs --port 6006")
    print("• Monitor: python monitor_training.py")
    print("• Resume training: python train_action_custom.py --resume latest_epoch.bin")

def plot_training_progress(checkpoint_dir='checkpoint_custom'):
    """Create training progress plots"""
    print("\n📈 Generating training plots...")
    
    # This would require parsing actual training logs
    # For now, create a placeholder plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.text(0.5, 0.5, 'No training data available\nStart training to see progress', 
             ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.subplot(1, 2, 2)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.text(0.5, 0.5, 'No training data available\nStart training to see progress', 
             ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("✅ Training plot saved as 'training_progress.png'")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        plot_training_progress()
    else:
        show_training_status() 