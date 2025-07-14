# MotionBERT: A Unified Perspective on Learning Human Motion Representations

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> [![arXiv](https://img.shields.io/badge/arXiv-2210.06551-b31b1b.svg)](https://arxiv.org/abs/2210.06551) <a href="https://motionbert.github.io/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a> <a href="https://youtu.be/slSPQ9hNLjM"><img alt="Demo" src="https://img.shields.io/badge/-Demo-ea3323?logo=youtube"></a> [![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffab41)](https://huggingface.co/walterzhu/MotionBERT)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motionbert-unified-pretraining-for-human/monocular-3d-human-pose-estimation-on-human3)](https://paperswithcode.com/sota/monocular-3d-human-pose-estimation-on-human3?p=motionbert-unified-pretraining-for-human)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motionbert-unified-pretraining-for-human/one-shot-3d-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/one-shot-3d-action-recognition-on-ntu-rgbd?p=motionbert-unified-pretraining-for-human)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motionbert-unified-pretraining-for-human/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=motionbert-unified-pretraining-for-human)

This is the official PyTorch implementation of the paper *"[MotionBERT: A Unified Perspective on Learning Human Motion Representations](https://arxiv.org/pdf/2210.06551.pdf)"* (ICCV 2023).

<img src="https://motionbert.github.io/assets/teaser.gif" alt="" style="zoom: 60%;" />

## Installation

```bash
conda create -n motionbert python=3.7 anaconda
conda activate motionbert
# Please install PyTorch according to your CUDA version.
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```



## Getting Started

| Task                              | Document                                                     |
| --------------------------------- | ------------------------------------------------------------ |
| Pretrain                          | [docs/pretrain.md](docs/pretrain.md)                                                          |
| 3D human pose estimation          | [docs/pose3d.md](docs/pose3d.md) |
| Skeleton-based action recognition | [docs/action.md](docs/action.md) |
| Mesh recovery                     | [docs/mesh.md](docs/mesh.md) |
| **MA-52 Micro-Action Recognition** | **See section below** |



## Micro-Action Recognition with MA-52 Dataset

This repository has been adapted to train MotionBERT on the **MA-52 (Micro-Action-52)** dataset for micro-action recognition. Micro-actions are imperceptible non-verbal behaviors characterized by low-intensity movements that offer insights into human feelings and intentions, making them valuable for emotion recognition and psychological assessment applications.

### 🎯 About MA-52 Dataset

The MA-52 dataset, developed by [VUT-HFUT](https://github.com/VUT-HFUT/Micro-Action), contains:
- **52 micro-action categories** covering comprehensive human micro-behaviors
- **22,422 video instances** from 205 participants
- **Whole-body perspective** including gestures, upper- and lower-limb movements
- **7 body part labels** for detailed analysis
- **Realistic psychological interview scenarios** providing natural micro-action data

### 🏷️ MA-52 Action Categories (52 Classes)

The dataset includes 52 distinct micro-action categories organized by body regions:

#### **Body Actions (A1-A6)**
- A1: Shaking body
- A2: Turning around  
- A3: Sitting straightly
- A4: Leaning forward
- A5: Leaning backward
- A6: Swaying body

#### **Head Actions (B1-B9)**
- B1: Nodding
- B2: Shaking head
- B3: Turning head
- B4: Tilting head
- B5: Lowering head
- B6: Raising head
- B7: Scratching head
- B8: Touching head
- B9: Supporting head

#### **Upper Limb Actions (C1-C20)**
- C1: Crossing arms
- C2: Folding arms
- C3: Waving hands
- C4: Clapping hands
- C5: Rubbing hands
- C6: Interlocking fingers
- C7: Covering face
- C8: Touching face
- C9: Touching chin
- C10: Touching nose
- C11: Touching mouth
- C12: Touching neck
- C13: Touching chest
- C14: Touching shoulder
- C15: Touching arm
- C16: Pointing
- C17: Gesturing
- C18: Adjusting clothes
- C19: Scratching
- C20: Fidgeting

#### **Lower Limb Actions (D1-D17)**
- D1: Crossing legs
- D2: Uncrossing legs
- D3: Stretching legs
- D4: Shaking legs
- D5: Tapping feet
- D6: Stamping feet
- D7: Moving feet
- D8: Adjusting sitting position
- D9: Touching thigh
- D10: Touching knee
- D11: Touching calf
- D12: Touching ankle
- D13: Touching foot
- D14: Adjusting shoes
- D15: Kicking
- D16: Stepping
- D17: Shifting weight

This comprehensive categorization enables detailed analysis of human micro-behaviors across different body regions for psychological and emotional assessment applications.

### 📁 MA-52 Data Structure

The MA-52 dataset should be organized as follows:

```
data/
├── train/
│   ├── A1_sample_001/
│   │   └── pose_3d.npy          # 3D pose data (T, 17, 3)
│   ├── B3_sample_002/
│   │   └── pose_3d.npy
│   ├── C5_sample_003/
│   │   └── pose_3d.npy
│   └── ...
├── test/
│   ├── A2_sample_100/
│   │   └── pose_3d.npy
│   └── ...
├── train_labels.txt              # Training labels (0-51 for 52 classes)
└── test_labels.txt               # Testing labels
```

### 📊 Data Format

#### **3D Pose Data (`pose_3d.npy`)**
- **Format**: NumPy array with shape `(T, 17, 3)`
- **T**: Number of frames (variable length, will be resampled to 243)
- **17**: Number of joints (H36M format)
- **3**: Coordinates (x, y, z in meters)

**H36M Joint Order:**
```python
joints = [
    'Pelvis',     'R_Hip',      'R_Knee',     'R_Ankle',    # 0-3
    'L_Hip',      'L_Knee',     'L_Ankle',    'Spine1',     # 4-7  
    'Neck',       'Head',       'Head_top',   'L_Shoulder', # 8-11
    'L_Elbow',    'L_Wrist',    'R_Shoulder', 'R_Elbow',    # 12-15
    'R_Wrist'                                               # 16
]
```

#### **Label Files (`train_labels.txt`, `test_labels.txt`)**
Each line contains: `folder_name class_id` (class_id ranges from 0-51 for 52 micro-actions)
```
A1_sample_001 0    # A1: Shaking body
B3_sample_002 11   # B3: Turning head  
C5_sample_003 17   # C5: Rubbing hands
D1_sample_004 26   # D1: Crossing legs
...
```

### 🏃‍♂️ Quick Start with MA-52

#### **1. Prepare MA-52 Data**
```bash
# Download MA-52 dataset from Hugging Face
# Visit: https://huggingface.co/datasets/kunli-cs/MA-52
# Or use the official application form for access

# Organize MA-52 dataset according to the structure above
mkdir -p data/train data/test

# Convert MA-52 video data to 3D pose format
python tools/convert_ma52_to_pose3d.py --input ma52_videos --output data/

# Verify data format
python verify_data_format.py
```

#### **2. Configure Training for MA-52**
Edit `configs/action/MB_train_custom.yaml`:
```yaml
# Data settings
data_root: data                    # Path to MA-52 data directory
action_classes: 52                 # MA-52 has 52 micro-action classes
clip_len: 243                      # Fixed sequence length for micro-actions

# Training settings
epochs: 300
batch_size: 24                     # Optimized for better GPU utilization
lr_backbone: 0.00001               # Lower learning rate for backbone
lr_head: 0.00008                   # Slightly higher learning rate for head
weight_decay: 0.1                  # Increased regularization
lr_decay: 0.98                     # More aggressive learning rate decay

# Model settings (Lighter architecture)
dim_feat: 192                      # Reduced feature dimension
depth: 2                           # Shallow transformer depth
dim_rep: 96                        # Reduced representation dimension
num_heads: 4                       # Fewer attention heads
hidden_dim: 64                     # Smaller hidden dimension
dropout_ratio: 0.7                 # Higher dropout for regularization

# Augmentation settings
random_move: True                  # Enable spatial augmentation
scale_range_train: [0.8, 3.2]     # Wider scale range for training
scale_range_test: [2, 2]          # Fixed scale for testing
normalize: True                    # Apply pose normalization
```

#### **3. Start Training on MA-52**
```bash
# Train MotionBERT on MA-52 dataset
python train_action_custom.py

# Resume training from checkpoint
python train_action_custom.py --resume latest_epoch.bin

# Train with specific config
python train_action_custom.py --config configs/action/MB_train_custom.yaml
```

#### **4. Monitor Training**
```bash
# View training status
python monitor_training.py

# Launch TensorBoard
tensorboard --logdir checkpoint_custom/logs --port 6006

# Generate training plots
python monitor_training.py plot
```

#### **5. Evaluate Model**
```bash
# Evaluate best checkpoint
python train_action_custom.py --evaluate best_epoch.bin

# Evaluate specific checkpoint
python train_action_custom.py --evaluate checkpoint_custom/epoch_100.bin
```

### 🏗️ Model Architecture

The current configuration uses a **lightweight MotionBERT architecture** optimized for faster training and inference:

- **Parameters**: ~2.8M (vs ~25M in full model)
- **Transformer Depth**: 2 layers (vs 5 in full model)
- **Feature Dimension**: 192 (vs 512 in full model)
- **Attention Heads**: 4 (vs 8 in full model)
- **Hidden Dimension**: 64 (vs 2048 in full model)
- **Dropout**: 0.7 (higher regularization)

This lighter architecture provides:
- ✅ **Faster training** (~3x speedup)
- ✅ **Lower memory usage** 
- ✅ **Good performance** on medium-scale datasets
- ✅ **Better regularization** with higher dropout

### 🔄 Data Augmentation Pipeline

The custom training includes comprehensive data augmentation:

#### **Training Augmentations:**
- **Pose Normalization**: Center around root joint, scale normalization
- **Temporal Resampling**: Random resampling to 243 frames
- **Spatial Transformations**: 
  - Random rotation: ±10 degrees
  - Random scaling: 90%-110%
  - Random translation: ±10%
- **Crop & Scale**: Normalize to [-1,1] + random scaling (0.8x-3.2x)

#### **Test Augmentations:**
- **Pose Normalization**: Same as training
- **Temporal Resampling**: Deterministic resampling
- **Crop & Scale**: Fixed 2x scaling
- **No Random Transformations**: Consistent evaluation

### 📈 Example Training Output on MA-52

```bash
INFO: Training samples: 17937 (MA-52 training set)
INFO: Test samples: 4485 (MA-52 test set)
INFO: Number of classes: 52 (MA-52 micro-actions)
INFO: Trainable parameter count: 2847628

Training epoch 0.
Train: [0][100/748]    BT 0.156 (0.164)    DT 0.089 (0.095)    
Loss 3.892 (3.945)    Acc@1 12.5 (10.8)    Acc@5 35.4 (32.1)

Test: [100/187]    Time 0.134 (0.142)    Loss 3.756 (3.801)    
Acc@1 15.2 (13.7)    Acc@5 38.9 (36.5)

Epoch 0 Summary:
  Train Loss: 3.945 | Train Acc@1: 10.8% | Train Acc@5: 32.1%
  Test Loss: 3.801 | Test Acc@1: 13.7% | Test Acc@5: 36.5%
  Best Acc@1: 13.7%
  Learning Rate: 0.000010
```

### 🎯 Performance Tips for MA-52 Micro-Actions

1. **Micro-Action Sensitivity**: MA-52 contains subtle micro-actions that require careful pose estimation quality
2. **Class Balance**: MA-52 has 52 fine-grained micro-action classes - ensure balanced training data
3. **Sequence Length**: 243 frames works well for capturing micro-action temporal dynamics
4. **Model Architecture**: Lightweight model (2.8M parameters) optimized for micro-action recognition
5. **Learning Rates**: Lower backbone LR (0.00001) with higher head LR (0.00008) for micro-action fine-tuning
6. **Regularization**: High dropout (0.7) and weight decay (0.1) prevent overfitting on similar micro-actions
7. **Batch Size**: Larger batch size (24) improves training stability with 52 micro-action classes
8. **Augmentation**: Wider scale range [0.8, 3.2] helps generalize across different body sizes and poses
9. **Body Region Focus**: Consider the 4 main body regions (Body, Head, Upper Limb, Lower Limb) during analysis
10. **Psychological Context**: MA-52 captures natural micro-behaviors from psychological interviews

### 🛠️ Troubleshooting

**Common Issues:**

```bash
# Check data format
python verify_data_format.py

# Validate your dataset
python lib/data/dataset_custom.py  # Run the test function

# Monitor GPU usage
watch nvidia-smi

# Check training progress
tail -f checkpoint_custom/logs/events.out.tfevents.*
```

**Data Preparation Scripts:**
- `tools/convert_ma52_to_pose3d.py`: Convert MA-52 video data to 3D pose format
- `verify_data_format.py`: Validate MA-52 data structure
- `lib/data/dataset_custom.py`: Custom dataset implementation for MA-52

**MA-52 Dataset Resources:**
- [MA-52 GitHub Repository](https://github.com/VUT-HFUT/Micro-Action)
- [MA-52 Dataset on Hugging Face](https://huggingface.co/datasets/kunli-cs/MA-52)
- [MA-52 Dataset Paper (TCSVT 2024)](https://doi.org/10.1109/TCSVT.2024.3358415)
- [VUT-HFUT Organization](https://github.com/VUT-HFUT)

## Applications

### In-the-wild inference (for custom videos)

Please refer to [docs/inference.md](docs/inference.md).

### Using MotionBERT for *human-centric* video representations

```python
'''	    
  x: 2D skeletons 
    type = <class 'torch.Tensor'>
    shape = [batch size * frames * joints(17) * channels(3)]
    
  MotionBERT: pretrained human motion encoder
    type = <class 'lib.model.DSTformer.DSTformer'>
    
  E: encoded motion representation
    type = <class 'torch.Tensor'>
    shape = [batch size * frames * joints(17) * channels(512)]
'''
E = MotionBERT.get_representation(x)
```



> **Hints**
>
> 1. The model could handle different input lengths (no more than 243 frames). No need to explicitly specify the input length elsewhere.
> 2. The model uses 17 body keypoints ([H36M format](https://github.com/JimmySuen/integral-human-pose/blob/master/pytorch_projects/common_pytorch/dataset/hm36.py#L32)). If you are using other formats, please convert them before feeding to MotionBERT. 
> 3. Please refer to [model_action.py](lib/model/model_action.py) and [model_mesh.py](lib/model/model_mesh.py) for examples of (easily) adapting MotionBERT to different downstream tasks.
> 4. For RGB videos, you need to extract 2D poses ([inference.md](docs/inference.md)), convert the keypoint format ([dataset_wild.py](lib/data/dataset_wild.py)), and then feed to MotionBERT ([infer_wild.py](infer_wild.py)).
>



## Model Zoo

| Model                           | Download Link                                                | Config                                                       | Performance      |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------- |
| MotionBERT (162MB)              | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgS425shtVi9e5reN?e=6UeBa2) | [pretrain/MB_pretrain.yaml](configs/pretrain/MB_pretrain.yaml) | -                |
| MotionBERT-Lite (61MB)          | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgS27Ydcbpxlkl0ng?e=rq2Btn) | [pretrain/MB_lite.yaml](configs/pretrain/MB_lite.yaml)       | -                |
| 3D Pose (H36M-SH, scratch)      | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgSvNejMQ0OHxMGZC?e=KcwBk1) | [pose3d/MB_train_h36m.yaml](configs/pose3d/MB_train_h36m.yaml) | 39.2mm (MPJPE)   |
| 3D Pose (H36M-SH, ft)           | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgSoTqtyR5Zsgi8_Z?e=rn4VJf) | [pose3d/MB_ft_h36m.yaml](configs/pose3d/MB_ft_h36m.yaml)     | 37.2mm (MPJPE)   |
| Action Recognition (x-sub, ft)  | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgTX23yT_NO7RiZz-?e=nX6w2j) | [action/MB_ft_NTU60_xsub.yaml](configs/action/MB_ft_NTU60_xsub.yaml) | 97.2% (Top1 Acc) |
| Action Recognition (x-view, ft) | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgTaNiXw2Nal-g37M?e=lSkE4T) | [action/MB_ft_NTU60_xview.yaml](configs/action/MB_ft_NTU60_xview.yaml) | 93.0% (Top1 Acc) |
| **MA-52 Micro-Actions (custom)** | [Google Drive](https://drive.google.com/file/d/1A18vkSfeDkkdkDEIOY8zWB0uA44xpjZO/view?usp=drive_link) | [action/MB_train_custom.yaml](configs/action/MB_train_custom.yaml) | **42.6% (Top1) / 79.9% (Top5)** |
| Mesh (with 3DPW, ft)            | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgTmgYNslCDWMNQi9?e=WjcB1F) | [mesh/MB_ft_pw3d.yaml](configs/mesh/MB_ft_pw3d.yaml)              | 88.1mm (MPVE)    |

In most use cases (especially with finetuning), `MotionBERT-Lite` gives a similar performance with lower computation overhead. 



## TODO

- [x] Scripts and docs for pretraining

- [x] Demo for custom videos



## Citation

If you find our work useful for your project, please consider citing the papers:

**MotionBERT:**
```bibtex
@inproceedings{motionbert2022,
  title     =   {MotionBERT: A Unified Perspective on Learning Human Motion Representations}, 
  author    =   {Zhu, Wentao and Ma, Xiaoxuan and Liu, Zhaoyang and Liu, Libin and Wu, Wayne and Wang, Yizhou},
  booktitle =   {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      =   {2023},
}
```

**MA-52 Dataset:**
```bibtex
@article{guo2024benchmarking,
  title={Benchmarking Micro-action Recognition: Dataset, Methods, and Applications},
  author={Guo, Dan and Li, Kun and Hu, Bin and Zhang, Yan and Wang, Meng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  volume={34},
  number={7},
  pages={6238-6252},
  publisher={IEEE},
  doi={10.1109/TCSVT.2024.3358415}
}

@article{li2024mmad,
  title={MMAD: Multi-label Micro-Action Detection in Videos},
  author={Li, Kun and Guo, Dan and Liu, Pengyu and Chen, Guoliang and Wang, Meng},
  journal={arXiv preprint arXiv:2407.05311},
  year={2024}
}

@misc{MicroAction2024,
  author       = {Guo, Dan and Li, Kun and Hu, Bin and Zhang, Yan and Wang, Meng},
  title        = {Micro-Action Benchmark},
  year         = {2024},
  howpublished = {\url{https://github.com/VUT-HFUT/Micro-Action}},
  note         = {Accessed: 2024-08-21}
}
```

