import torch
import numpy as np
import os
import random
import copy
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale, resample
from lib.utils.tools import read_pkl

def random_move(data_numpy,
                angle_range=[-10., 10.],
                scale_range=[0.9, 1.1],
                transform_range=[-0.1, 0.1],
                move_time_candidate=[1]):
    """
    Apply random augmentation to pose data
    Input: data_numpy (M, T, V, C) where M=persons, T=frames, V=joints, C=coordinates
    """
    data_numpy = np.transpose(data_numpy, (3,1,2,0)) # M,T,V,C-> C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)
    A = np.random.uniform(angle_range[0], angle_range[1], num_node)
    S = np.random.uniform(scale_range[0], scale_range[1], num_node)
    T_x = np.random.uniform(transform_range[0], transform_range[1], num_node)
    T_y = np.random.uniform(transform_range[0], transform_range[1], num_node)
    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)
    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])
    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])
    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)
    data_numpy = np.transpose(data_numpy, (3,1,2,0)) # C,T,V,M -> M,T,V,C
    return data_numpy    

def load_labels(label_file):
    """
    Load labels from text file
    Format: filename.mp4 label_number
    """
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[0].replace('.mp4', '')  # Remove .mp4 extension
                label = int(parts[1])
                labels[filename] = label
    return labels

def normalize_pose_data(pose_data):
    """
    Normalize 3D pose data to be more suitable for training
    Input: pose_data (T, 17, 3) - Time, Joints, Coordinates
    Output: normalized pose_data
    """
    # Center the pose around the root joint (assuming joint 0 is root)
    root_joint = pose_data[:, 0:1, :]  # (T, 1, 3)
    pose_data_centered = pose_data - root_joint
    
    # Scale based on the overall pose size
    pose_scale = np.std(pose_data_centered.reshape(-1, 3), axis=0)
    pose_scale = np.maximum(pose_scale, 1e-8)  # Avoid division by zero
    pose_data_normalized = pose_data_centered / pose_scale
    
    return pose_data_normalized

def convert_to_motionbert_format(pose_data):
    """
    Convert 3D pose data to MotionBERT format
    Input: pose_data (T, 17, 3) - assumes H36M joint order
    Output: pose_data in MotionBERT expected format (M, T, V, C)
    """
    # Assuming single person data, expand to M=2 format (person + fake zero person)
    T, V, C = pose_data.shape
    
    # Create motion array with 2 persons (M=2)
    motion = np.zeros((2, T, V, C))
    motion[0] = pose_data  # Real person
    motion[1] = np.zeros((T, V, C))  # Fake zero person
    
    return motion.astype(np.float32)

class CustomActionDataset(Dataset):
    def __init__(self, data_root, data_split, n_frames=243, random_move=True, scale_range=[1,1], 
                 normalize=True, action_classes=None):
        """
        Custom dataset for 3D pose data
        
        Args:
            data_root: Root directory containing train/test folders
            data_split: 'train' or 'test'
            n_frames: Target number of frames (243 for MotionBERT)
            random_move: Apply random augmentation
            scale_range: Scale range for augmentation
            normalize: Whether to normalize pose data
            action_classes: Number of action classes (auto-detected if None)
        """
        np.random.seed(0)
        
        self.data_root = data_root
        self.data_split = data_split
        self.n_frames = n_frames
        self.random_move = random_move
        self.scale_range = scale_range
        self.normalize = normalize
        self.is_train = data_split == 'train'
        
        # Load labels
        label_file = os.path.join(data_root, f'{data_split}_labels.txt')
        self.labels_dict = load_labels(label_file)
        
        # Load data
        self.motions = []
        self.labels = []
        
        data_dir = os.path.join(data_root, data_split)
        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            pose_file = os.path.join(folder_path, 'pose_3d.npy')
            if not os.path.exists(pose_file):
                continue
                
            # Load pose data
            pose_data = np.load(pose_file)  # (T, 17, 3)
            
            # Normalize if requested
            if self.normalize:
                pose_data = normalize_pose_data(pose_data)
            
            # Resample to target length
            T = pose_data.shape[0]
            resample_id = resample(ori_len=T, target_len=n_frames, randomness=self.is_train)
            pose_data_resampled = pose_data[resample_id]
            
            # Convert to MotionBERT format
            motion = convert_to_motionbert_format(pose_data_resampled)
            
            # Get label
            if folder_name in self.labels_dict:
                label = self.labels_dict[folder_name]
            else:
                print(f"Warning: No label found for {folder_name}")
                continue
                
            self.motions.append(motion)
            self.labels.append(label)
        
        self.motions = np.array(self.motions)
        self.labels = np.array(self.labels)
        
        # Auto-detect number of classes if not provided
        if action_classes is None:
            self.action_classes = len(np.unique(self.labels))
        else:
            self.action_classes = action_classes
        
        print(f"Loaded {len(self.motions)} samples from {data_split} split")
        print(f"Motion shape: {self.motions.shape}")
        print(f"Number of classes: {self.action_classes}")
        print(f"Labels range: {self.labels.min()} - {self.labels.max()}")
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions)

    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, label = self.motions[idx], self.labels[idx] # (M,T,J,C)
        
        # Apply random augmentation if training
        if self.random_move and self.is_train:
            motion = random_move(motion)
        
        # Apply scaling if requested
        if self.scale_range and self.scale_range != [1, 1]:
            motion = crop_scale(motion, scale_range=self.scale_range)
        
        return motion.astype(np.float32), label

def test_custom_dataset():
    """Test function to verify the custom dataset works correctly"""
    data_root = "data"
    
    # Test train dataset
    train_dataset = CustomActionDataset(
        data_root=data_root,
        data_split='train',
        n_frames=243,
        random_move=True,
        scale_range=[1, 3],
        normalize=True
    )
    
    # Test test dataset  
    test_dataset = CustomActionDataset(
        data_root=data_root,
        data_split='test',
        n_frames=243,
        random_move=False,
        scale_range=[2, 2],
        normalize=True
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Test data loading
    train_sample = train_dataset[0]
    test_sample = test_dataset[0]
    
    print(f"Train sample shape: {train_sample[0].shape}, label: {train_sample[1]}")
    print(f"Test sample shape: {test_sample[0].shape}, label: {test_sample[1]}")
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    test_custom_dataset() 