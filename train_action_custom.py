import os
import numpy as np
import argparse
import errno
import math
import pickle
import tensorboardX
from tqdm import tqdm
from time import time
import copy
import random
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from functools import partial

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_custom import CustomActionDataset
from lib.model.loss import *
from lib.model.model_action import ActionNet
from lib.model.DSTformer import DSTformer

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/action/MB_train_custom.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint_custom', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100, type=int, help='print frequency')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    opts = parser.parse_args()
    return opts

def validate(test_loader, model, criterion, opts):
    """Separate validation function with proper metrics tracking"""
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        end = time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader), desc="Validating"):
            batch_size = len(batch_input)
            
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            
            output = model(batch_input)    # (N, num_classes)
            loss = criterion(output, batch_gt)

            # Update metrics
            losses.update(loss.item(), batch_size)
            acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # Measure elapsed time
            batch_time.update(time() - end)
            end = time()

            if (idx + 1) % opts.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       idx + 1, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))
    
    return losses.avg, top1.avg, top5.avg

def train_with_config(args, opts):
    print("Training configuration:")
    print(args)
    
    # Create checkpoint directory
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    
    # Setup TensorBoard logging
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    
    # Create backbone
    backbone = DSTformer(
        dim_in=3, 
        dim_out=3, 
        dim_feat=args.dim_feat, 
        dim_rep=args.dim_rep,
        depth=args.depth, 
        num_heads=args.num_heads, 
        mlp_ratio=args.mlp_ratio, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        maxlen=args.maxlen, 
        num_joints=args.num_joints
    )
    
    # Handle finetune mode
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            backbone = load_pretrained_weights(backbone, checkpoint)
    
    # Handle partial training
    if args.partial_train:
        backbone = partial_train_layers(backbone, args.partial_train)
    
    # Create full model
    model = ActionNet(
        backbone=backbone,
        dim_rep=args.dim_rep,
        num_classes=args.action_classes,
        dropout_ratio=args.dropout_ratio,
        version=args.model_version,
        hidden_dim=args.hidden_dim,
        num_joints=args.num_joints
    )
    
    # Setup loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
    
    # Count model parameters
    best_acc = 0
    model_params = sum(p.numel() for p in model.parameters())
    print('INFO: Trainable parameter count:', model_params)
    
    # Load datasets
    print('Loading dataset...')
    
    # Optimized DataLoader parameters
    trainloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }
    
    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }
    
    # Create custom datasets
    train_dataset = CustomActionDataset(
        data_root=args.data_root,
        data_split='train',
        n_frames=args.clip_len,
        random_move=args.random_move,
        scale_range=args.scale_range_train,
        normalize=args.normalize
    )
    
    test_dataset = CustomActionDataset(
        data_root=args.data_root,
        data_split='test',
        n_frames=args.clip_len,
        random_move=False,
        scale_range=args.scale_range_test,
        normalize=args.normalize
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)
    
    print(f'INFO: Training samples: {len(train_dataset)}')
    print(f'INFO: Test samples: {len(test_dataset)}')
    print(f'INFO: Number of classes: {args.action_classes}')
    
    # Check for existing checkpoint
    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    
    # Load checkpoint if resuming or evaluating
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
        
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
    
    # Skip training if only evaluating
    if opts.evaluate:
        test_loss, test_top1, test_top5 = validate(test_loader, model, criterion, opts)
        print('Evaluation Results:')
        print('Loss {loss:.4f} \t'
              'Acc@1 {top1:.3f} \t'
              'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))
        return
    
    # Setup optimizer with different learning rates for backbone and head
    optimizer = optim.AdamW([
        {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
        {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
    ], lr=args.lr_backbone, weight_decay=args.weight_decay)
    
    # Setup learning rate scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    
    # Setup training start epoch
    start_epoch = 0
    print('INFO: Training on {} batches'.format(len(train_loader)))
    
    # Resume training state if available
    if opts.resume:
        start_epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        
        if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f'Training epoch {epoch}.')
        
        # Training metrics
        losses_train = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        # Set model to training mode
        model.train()
        end = time()
        
        # Training iteration
        for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}"):
            data_time.update(time() - end)
            batch_size = len(batch_input)
            
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            
            # Forward pass
            output = model(batch_input)  # (N, num_classes)
            
            # Calculate loss
            optimizer.zero_grad()
            loss_train = criterion(output, batch_gt)
            
            # Update training metrics
            losses_train.update(loss_train.item(), batch_size)
            acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            
            # Backward pass
            loss_train.backward()
            optimizer.step()
            
            # Measure elapsed time
            batch_time.update(time() - end)
            end = time()
            
            # Print progress
            if (idx + 1) % opts.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses_train, top1=top1, top5=top5))
                sys.stdout.flush()
        
        # Validation
        test_loss, test_top1, test_top5 = validate(test_loader, model, criterion, opts)
        
        # Log to TensorBoard
        train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
        train_writer.add_scalar('train_top1', top1.avg, epoch + 1)
        train_writer.add_scalar('train_top5', top5.avg, epoch + 1)
        train_writer.add_scalar('test_loss', test_loss, epoch + 1)
        train_writer.add_scalar('test_top1', test_top1, epoch + 1)
        train_writer.add_scalar('test_top5', test_top5, epoch + 1)
        train_writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch + 1)
        
        # Update learning rate
        scheduler.step()
        
        # Save latest checkpoint
        chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')
        print('Saving checkpoint to', chk_path)
        torch.save({
            'epoch': epoch + 1,
            'lr': scheduler.get_last_lr(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'model': model.state_dict(),
            'best_acc': best_acc
        }, chk_path)
        
        # Save best checkpoint
        if test_top1 > best_acc:
            best_acc = test_top1
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin')
            print(f"New best accuracy: {best_acc:.3f}% - Saving best checkpoint")
            torch.save({
                'epoch': epoch + 1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'model': model.state_dict(),
                'best_acc': best_acc
            }, best_chk_path)
        
        # Print epoch summary
        print(f'Epoch {epoch} Summary:')
        print(f'  Train Loss: {losses_train.avg:.4f} | Train Acc@1: {top1.avg:.3f}% | Train Acc@5: {top5.avg:.3f}%')
        print(f'  Test Loss: {test_loss:.4f} | Test Acc@1: {test_top1:.3f}% | Test Acc@5: {test_top5:.3f}%')
        print(f'  Best Acc@1: {best_acc:.3f}%')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 80)
    
    print('Training completed!')
    train_writer.close()

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts) 