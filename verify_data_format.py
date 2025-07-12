#!/usr/bin/env python3
"""
Data format verification script for MotionBERT custom dataset
This script checks if your data is in the correct format without requiring torch
"""

import numpy as np
import os
import sys

def check_data_format():
    """Check if the data format is correct"""
    print("🔍 Checking data format...")
    
    # Check directory structure
    data_root = "data"
    required_files = [
        "train_labels.txt",
        "test_labels.txt",
        "train",
        "test"
    ]
    
    for file in required_files:
        path = os.path.join(data_root, file)
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
            return False
        else:
            print(f"✅ Found: {path}")
    
    # Check labels format
    print("\n📋 Checking labels format...")
    for split in ['train', 'test']:
        label_file = os.path.join(data_root, f"{split}_labels.txt")
        with open(label_file, 'r') as f:
            lines = f.readlines()[:5]  # Check first 5 lines
            print(f"  {split} labels sample:")
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    label = parts[1]
                    print(f"    {filename} -> {label}")
                else:
                    print(f"    ⚠️ Invalid line: {line.strip()}")
    
    # Check pose data format
    print("\n🦴 Checking pose data format...")
    train_dir = os.path.join(data_root, "train")
    sample_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))][:3]
    
    for folder in sample_folders:
        folder_path = os.path.join(train_dir, folder)
        pose_file = os.path.join(folder_path, "pose_3d.npy")
        
        if os.path.exists(pose_file):
            try:
                pose_data = np.load(pose_file)
                print(f"  ✅ {folder}/pose_3d.npy: {pose_data.shape}, dtype: {pose_data.dtype}")
                
                # Check if shape is correct
                if len(pose_data.shape) == 3 and pose_data.shape[1] == 17 and pose_data.shape[2] == 3:
                    print(f"    ✅ Correct shape: (T={pose_data.shape[0]}, V=17, C=3)")
                else:
                    print(f"    ❌ Incorrect shape: expected (T, 17, 3), got {pose_data.shape}")
                    
                # Check data range
                print(f"    📊 Data range: [{pose_data.min():.3f}, {pose_data.max():.3f}]")
                
            except Exception as e:
                print(f"  ❌ Error loading {folder}/pose_3d.npy: {e}")
        else:
            print(f"  ❌ Missing: {folder}/pose_3d.npy")
    
    # Count samples
    print("\n📊 Dataset statistics:")
    for split in ['train', 'test']:
        split_dir = os.path.join(data_root, split)
        folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
        pose_files = 0
        for folder in folders:
            pose_file = os.path.join(split_dir, folder, "pose_3d.npy")
            if os.path.exists(pose_file):
                pose_files += 1
        print(f"  {split}: {pose_files} samples")
    
    # Check label consistency
    print("\n🔗 Checking label consistency...")
    for split in ['train', 'test']:
        label_file = os.path.join(data_root, f"{split}_labels.txt")
        split_dir = os.path.join(data_root, split)
        
        # Get labels
        labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0].replace('.mp4', '')
                    label = int(parts[1])
                    labels[filename] = label
        
        # Get folders
        folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
        
        # Check consistency
        labeled_folders = set(labels.keys())
        actual_folders = set(folders)
        
        print(f"  {split}:")
        print(f"    Labeled samples: {len(labeled_folders)}")
        print(f"    Actual folders: {len(actual_folders)}")
        print(f"    Missing labels: {len(actual_folders - labeled_folders)}")
        print(f"    Extra labels: {len(labeled_folders - actual_folders)}")
        
        if labeled_folders == actual_folders:
            print(f"    ✅ Perfect match!")
        else:
            print(f"    ⚠️ Mismatch detected")
    
    print("\n" + "="*50)
    print("✅ Data format verification complete!")
    print("📖 See PREPROCESSING_GUIDE.md for next steps")
    
    return True

if __name__ == "__main__":
    check_data_format() 