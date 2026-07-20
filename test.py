#!/usr/bin/env python3
"""
Distributed training script for ResNet-50 variants.
Supports:
1. CIFAR-100
2. ImageNet (ImageNet-1K directory layout with train/ and val/)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
import time
import os
import argparse
from datetime import datetime, timedelta

from fvcore.nn import FlopCountAnalysis, parameter_count_table

from models_unified import (
    MoCE as UnifiedMoCE,
    resnet50_baseline,
    resnet101_baseline,
    resnet152_baseline,
    resnet50_moce_projection,
    resnet101_moce_projection,
    resnet152_moce_projection,
)
import sys

class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def isatty(self):
        return getattr(self.stdout, "isatty", lambda: False)()


def is_moce_module(module):
    return isinstance(module, UnifiedMoCE) or (
        module.__class__.__name__ == "MoCE"
        and hasattr(module, "routing_logits")
        and hasattr(module, "gate")
        and hasattr(module, "aux_loss")
    )

 
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def compute_model_flops_params(model, device, args, num_classes):
    """
    Returns (flops, params) for a single forward with batch=1.
    FLOPs are for the model forward only (not backward).
    """
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()

    if args.dataset == "cifar100":
        dummy = torch.randn(1, 3, 32, 32, device=device)
    elif args.dataset == "imagenet":
        dummy = torch.randn(1, 3, 224, 224, device=device)
    else:
        raise ValueError(args.dataset)

    with torch.no_grad():
        flops = FlopCountAnalysis(base_model, dummy).total()
    params = sum(p.numel() for p in base_model.parameters())

    return flops, params


def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=30)
    )


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def dataset_num_classes(dataset_name):
    """Return number of classes for supported datasets."""
    if dataset_name == 'cifar100':
        return 100
    if dataset_name == 'imagenet':
        return 1000
    raise ValueError(f"Unsupported dataset '{dataset_name}'.")


def get_dataloader(batch_size, rank, world_size, args):
    """Create train/val dataloaders for CIFAR-100 or ImageNet."""
    if args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])

        trainset = torchvision.datasets.CIFAR100(
            root=args.cifar_root, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root=args.cifar_root, train=False, download=True, transform=transform_test
        )
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.imagenet_root, 'train')
        val_dir = os.path.join(args.imagenet_root, 'val')
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise FileNotFoundError(
                "ImageNet dataset not found. Expected directories:\n"
                f"  {train_dir}\n"
                f"  {val_dir}"
            )

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(val_dir, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset '{args.dataset}'.")

    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=rank, shuffle=False)

    persistent_workers = args.num_workers > 0
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    return trainloader, testloader, train_sampler


def collect_aux_loss(model):
    """Collect auxiliary balance losses from all MoCE modules."""
    aux_loss = None
    # Unwrap DDP if needed
    base_model = model.module if hasattr(model, 'module') else model
    for m in base_model.modules():
        if is_moce_module(m):
            module_aux_loss = m.aux_loss if isinstance(m.aux_loss, torch.Tensor) else torch.as_tensor(
                m.aux_loss, device=next(m.parameters()).device
            )
            aux_loss = module_aux_loss if aux_loss is None else aux_loss + module_aux_loss
            # print(f"Collected aux_loss from MoCE module: {module_aux_loss.item():.4f}")
    if aux_loss is None:
        return torch.tensor(0.0, device=next(base_model.parameters()).device)
    return aux_loss


def apply_moce_loss_lambdas(model, args):
    """Apply MoCE aux-loss lambdas from CLI to all MoCE modules."""
    base_model = model.module if hasattr(model, 'module') else model
    for m in base_model.modules():
        if is_moce_module(m):
            m.balance_weight = args.moce_balance_weight
            m.specialization_weight = args.moce_specialization_weight
            m.diversity_weight = args.moce_diversity_weight
            # print(f"Applied MoCE loss lambdas: bal={m.balance_weight}, spe={m.specialization_weight}, div={m.diversity_weight}")

def build_optimizer(model, base_lr, momentum=0.9, weight_decay=5e-4, router_lr_mult=1.0):
    """
    Build SGD optimizer with separate param groups for routing/gate params.

    Routing logits and gate are kept at zero weight decay to avoid
    collapsing logits toward uniform values.
    """
    base_model = model.module if hasattr(model, 'module') else model
    router_params = []
    router_param_ids = set()

    for module in base_model.modules():
        if is_moce_module(module):
            router_params.append(module.routing_logits)
            router_param_ids.add(id(module.routing_logits))
            for p in module.gate.parameters():
                router_params.append(p)
                router_param_ids.add(id(p))

    main_params = [p for p in base_model.parameters() if id(p) not in router_param_ids]

    param_groups = []
    if main_params:
        param_groups.append({
            'params': main_params,
            'lr': base_lr,
            'weight_decay': weight_decay
        })
    if router_params:
        param_groups.append({
            'params': router_params,
            'lr': base_lr * router_lr_mult,
            'weight_decay': 0.0
        })

    return optim.SGD(param_groups, momentum=momentum)


def resolve_scheduler_config(args):
    """Resolve automatic scheduler/warmup defaults."""
    if args.scheduler == 'auto':
        args.scheduler = 'cosine'

    if args.warmup_epochs is None:
        args.warmup_epochs = 5 if args.dataset == 'imagenet' else 0

    if args.warmup_epochs < 0:
        raise ValueError(f"warmup-epochs must be >= 0, got {args.warmup_epochs}.")
    if not (0.0 < args.warmup_start_factor <= 1.0):
        raise ValueError(
            f"warmup-start-factor must be in (0, 1], got {args.warmup_start_factor}."
        )
    if args.step_size < 1:
        raise ValueError(f"step-size must be >= 1, got {args.step_size}.")


def build_scheduler(optimizer, args):
    """Build scheduler, optionally with linear warmup."""
    if args.scheduler == 'cosine':
        main_epochs = max(1, args.epochs - args.warmup_epochs)
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=args.min_lr
        )
    elif args.scheduler == 'step':
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    else:
        raise ValueError(f"Unsupported scheduler '{args.scheduler}'.")

    if args.warmup_epochs <= 0:
        return main_scheduler, args.scheduler

    warmup_epochs = min(args.warmup_epochs, args.epochs)
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    if warmup_epochs >= args.epochs:
        return warmup, f"linear_warmup({warmup_epochs})"

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, main_scheduler],
        milestones=[warmup_epochs]
    )
    return scheduler, f"linear_warmup({warmup_epochs})+{args.scheduler}"


def train_epoch(model, trainloader, criterion, optimizer, device, scaler=None):
    """Train one epoch."""
    model.train()
    correct = 0
    total = 0
    total_ce_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            ce_loss = criterion(outputs, targets)

            # Add auxiliary balance loss from MoCE modules
            aux_loss = collect_aux_loss(model)
            if isinstance(aux_loss, torch.Tensor):
                loss = ce_loss + aux_loss
                total_aux_loss += aux_loss.item()
            else:
                loss = ce_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_ce_loss += ce_loss.item()
        num_batches += 1
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_ce = total_ce_loss / num_batches
    avg_aux = total_aux_loss / num_batches
    acc = 100. * correct / total
    return acc, avg_ce, avg_aux

def build_moce_cache(model):
    m = model.module if hasattr(model, "module") else model
    for module in m.modules():
        if hasattr(module, "build_eval_cache"):
            module.build_eval_cache()
            
def test(model, testloader, device, world_size, scaler=None):
    """Test the model with mixed precision support."""
    model.eval()
    build_moce_cache(model)
    
    correct = 0
    total = 0
    # Determine if we should use AMP based on whether a scaler was provided
    # or if you prefer a hardcoded flag.
    use_amp = scaler is not None

    # We use no_grad for evaluation to save memory and compute
    with torch.no_grad():
        # Match the training loop's autocast logic
        with torch.cuda.amp.autocast(enabled=use_amp):
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                outputs = model(inputs)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
    # --- Distributed Reduction ---
    # Gather results from all GPUs to calculate the global accuracy
    correct_tensor = torch.tensor([correct], dtype=torch.float32, device=device)
    total_tensor = torch.tensor([total], dtype=torch.float32, device=device)
    
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    # Calculate final accuracy across all ranks
    final_total = total_tensor.item()
    acc = 100. * correct_tensor.item() / final_total if final_total > 0 else 0.0
    
    return acc


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, model_name, trainloader, testloader, train_sampler, 
                device, world_size, rank, args,run_dir, last_ckpt_path, best_ckpt_path):
    """Train a single model configuration."""

    # Optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        base_lr=args.lr * world_size,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        router_lr_mult=args.router_lr_mult
    )
    scheduler, scheduler_name = build_scheduler(optimizer, args)
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Training: {model_name}")
        print(f"{'='*80}")
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Dataset: {args.dataset}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size} ({args.batch_size // world_size} per GPU)")
        print(f"Scheduler: {scheduler_name}")
        print(f"Warmup epochs: {args.warmup_epochs}")
        print(f"{'='*80}\n")
    
    # Training loop
    best_acc = 0
    best_acc_epoch = 0
    start_epoch = 0
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Apply MoCE lambdas once before training loop
    if 'moce' in args.model:
        apply_moce_loss_lambdas(model, args)
        if rank == 0:
            print(
                f"MoCE loss lambdas for this run: "
                f"bal={args.moce_balance_weight}, "
                f"spe={args.moce_specialization_weight}, "
                f"div={args.moce_diversity_weight}"
            )

    if os.path.exists(last_ckpt_path):
        if rank == 0:
            print(f"Resuming from checkpoint: {last_ckpt_path}")
        checkpoint = torch.load(last_ckpt_path, map_location=device)

        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        best_acc_epoch = checkpoint.get('best_acc_epoch', 0)
        # scaler.load_state_dict(checkpoint['scaler'])
        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])

    start_time_total = time.time()
    
    # Remove old balance-only warmup logic for new composite loss
    # balance_weight_target = args.balance_weight

    # Print initial routing state
    # if 'moce' in args.model:
    #     print("lolkolololo")
    #     sample_batch = next(iter(trainloader))[0][:8].to(device)
    #     if rank == 0:
    #         from MoCE import print_routing_diagnostics
    #         # print_routing_diagnostics(model, sample_input=sample_batch, header="Initial (before training)")
    #         print_routing_diagnostics(model,header=f"Initial Routing Diagnostics - {model_name}")
    # sample_batch = next(iter(trainloader))[0][:8].to(device)
    
    for epoch in range(start_epoch,args.epochs):
        train_sampler.set_epoch(epoch)

        # Keep lambdas fixed (or CLI-controlled) across epochs
        if 'moce' in args.model:
            apply_moce_loss_lambdas(model, args)

        # epoch_start = time.time()
        
        # Train
        train_start = time.time()
        train_acc, ce_loss, aux_loss = train_epoch(model, trainloader, criterion, optimizer, device, scaler)
        train_time = time.time() - train_start
        
        # Test
        infer_start = time.time()
        test_acc = test(model, testloader, device, world_size, scaler)
        infer_time = time.time() - infer_start        
        
        # Calculate best
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_epoch = epoch
        
        # epoch_time = time.time() - epoch_start
        
        # Print
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            epoch_time = train_time + infer_time

            print(f"[{now}] Epoch {epoch+1:3d}/{args.epochs} | {model_name:15s} | "
                f"Epoch: {epoch_time:5.1f}s | Train: {train_time:5.1f}s | Infer: {infer_time:5.1f}s | "
                f"Acc: {test_acc:6.2f}% | "
                f"Best: {best_acc:6.2f}% (Epoch {best_acc_epoch + 1}) | "
                f"CE: {ce_loss:.4f} | Aux: {aux_loss:.4f} | "
                f"bal={args.moce_balance_weight:.4f} | "
                f"spe={args.moce_specialization_weight:.4f} | "
                f"div={args.moce_diversity_weight:.4f} | "
                f"LR: {current_lr:.6f}")
        
        # Routing diagnostics every 25 epochs + first few epochs
        # if rank == 0 and 'moce' in args.model:
        #     if epoch < 5 or (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
        #         from MoCE import print_routing_diagnostics
        #         # print_routing_diagnostics(model, sample_input=sample_batch, header=f"Epoch {epoch+1}/{args.epochs}")
        #         print_routing_diagnostics(model,header=f"Routing Diagnostics - Epoch {epoch+1}/{args.epochs} - {model_name}")

        if rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
                'best_acc_epoch': best_acc_epoch,
                'scaler': scaler.state_dict() if scaler is not None else None
            }

            torch.save(checkpoint, last_ckpt_path)

            if test_acc >= best_acc:
                torch.save(checkpoint, best_ckpt_path)
        
        scheduler.step()
        
    
    total_time = time.time() - start_time_total
    
    # Final summary
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Results: {model_name}")
        print(f"{'='*80}")
        print(f"Best Accuracy: {best_acc:.2f}% (Epoch {best_acc_epoch + 1})")
        print(f"Total Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"Time per Epoch: {total_time/args.epochs:.1f}s")
        print(f"{'='*80}\n")
    
    return {
        'model_name': model_name,
        'best_acc': best_acc,
        'best_acc_epoch': best_acc_epoch + 1,
        'total_time': total_time,
        'time_per_epoch': total_time / args.epochs,
        'parameters': count_parameters(model)
    }


def _layers_config_for_model(model_name: str):
    if model_name == "moce50":
        return [3, 4, 6, 3]
    if model_name == "moce101":
        return [3, 4, 23, 3]
    if model_name == "moce152":
        return [3, 8, 36, 3]
    return None


def parse_k_config_arg(k_config: str, layers_config):
    """
    Parse k config from CLI.
    Format (4 stages):
      - stage scalar: "8|12|24|48" (expanded per block)
      - per-block:    "8,8,8|12,12,12,12|24,24,24,24,24,24|48,48,48"
      - mixed allowed
    """
    stage_tokens = [t.strip() for t in k_config.split("|")]
    if len(stage_tokens) != len(layers_config):
        raise ValueError(
            f"--k-config must have {len(layers_config)} stages separated by '|', "
            f"got {len(stage_tokens)}"
        )

    out = []
    for i, (tok, n_blocks) in enumerate(zip(stage_tokens, layers_config)):
        if "," in tok:
            vals = [int(x.strip()) for x in tok.split(",") if x.strip() != ""]
            if len(vals) != n_blocks:
                raise ValueError(
                    f"Stage {i+1} needs {n_blocks} values, got {len(vals)} in '{tok}'"
                )
            out.append(vals)
        else:
            v = int(tok)
            out.append([v] * n_blocks)
    return out

def main():
    parser = argparse.ArgumentParser(description='Distributed ResNet-50 training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=196, help='Total batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Base learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--router-lr-mult', type=float, default=1,
                        help='Multiplier for router parameter group LR')
    parser.add_argument('--balance-weight', type=float, default=0.005,
                        help='Auxiliary load-balancing loss weight for MoCE modules')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'imagenet'],
                        help='Dataset to train on')
    parser.add_argument('--cifar-root', type=str, default='/data',
                        help='Root directory for CIFAR-100')
    parser.add_argument('--imagenet-root', type=str, default='/data/imagenet',
                        help='ImageNet root containing train/ and val/')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader worker count')
    parser.add_argument('--scheduler', type=str, default='auto',
                        choices=['auto', 'cosine', 'step'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=None,
                        help='Linear warmup epochs. Auto: 0 for CIFAR-100, 5 for ImageNet')
    parser.add_argument('--warmup-start-factor', type=float, default=0.1,
                        help='Warmup starting LR multiplier')
    parser.add_argument('--min-lr', type=float, default=0.0,
                        help='Minimum LR for cosine scheduler')
    parser.add_argument('--step-size', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    parser.add_argument('--model',type=str,default='moce50',
        choices=[
            'baseline50', 'baseline101', 'baseline152',
            'moce50', 'moce101', 'moce152'
        ],
        help='Model variant to train'
    )
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable mixed-precision training (AMP)')
    parser.add_argument('--run-name', type=str, required=True,help='Name of the experiment run')
    parser.add_argument('--k', type=int, default=8, help='Number of experts (k) for MoCE layers')

    # New MoCE composite-loss lambdas (defaults from MoCE.py header)
    parser.add_argument('--moce-balance-weight', type=float, default=0.05,
                        help='MoCE lambda for balance term')
    parser.add_argument('--moce-specialization-weight', type=float, default=0.1,
                        help='MoCE lambda for specialization term')
    parser.add_argument('--moce-diversity-weight', type=float, default=0.01,
                        help='MoCE lambda for diversity term')

    # New per-layer/per-block k override
    parser.add_argument(
        '--k-config',
        type=str,
        default=None,
        help=("Per-stage/per-block k config. "
              "Example stage-wise: '8|12|24|48' "
              "Example per-block (r50): '8,8,8|12,12,12,12|24,24,24,24,24,24|48,48,48'")
    )

    args = parser.parse_args()

    # Backward-compat handling
    # if args.balance_weight is not None:
    #     args.moce_balance_weight = args.balance_weight

    checkpoint_root = "checkpoints"
    os.makedirs(checkpoint_root, exist_ok=True)

    run_dir = os.path.join(checkpoint_root, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_file_path = os.path.join(run_dir, "train.log")

    last_ckpt_path = os.path.join(run_dir, "last.pth")
    best_ckpt_path = os.path.join(run_dir, "best.pth")

    resolve_scheduler_config(args)
    
    # Enable cuDNN optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Get rank and world size
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if args.batch_size % world_size != 0:
        raise ValueError(
            f"batch-size ({args.batch_size}) must be divisible by world_size ({world_size})."
        )
    
    # Setup distributed
    setup_distributed(rank, world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    if rank == 0:
        log_file_path = os.path.join(run_dir, "train.log")
        sys.stdout = Tee(log_file_path)
        sys.stderr = sys.stdout

    moce_k = None
    model_layers_cfg = _layers_config_for_model(args.model)
    if model_layers_cfg is not None:
        if args.k_config is not None:
            moce_k = parse_k_config_arg(args.k_config, model_layers_cfg)
        elif args.k is not None:
            moce_k = args.k

    if rank == 0:
        print(f"  Model: {args.model}")
        if model_layers_cfg is not None:
            if args.k_config is not None:
                print(f"  MoCE k-config (CLI): {args.k_config}")
            elif args.k is not None:
                print(f"  MoCE k (global CLI): {args.k}")
            else:
                print(f"  MoCE k: model defaults from models_unified.py")
        print(f"  AMP: {args.amp}")
    
    scaled_lr = args.lr * world_size
    if args.warmup_epochs > 0:
        scheduler_display = f"linear_warmup({args.warmup_epochs})+{args.scheduler}"
    else:
        scheduler_display = args.scheduler

    # Print header
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Distributed ResNet Training")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch Size: {args.batch_size} ({args.batch_size // world_size} per GPU)")
        print(f"  Learning Rate: {args.lr} -> Scaled: {scaled_lr}")
        print(f"  Optimizer: SGD (momentum={args.momentum}, weight_decay={args.weight_decay})")
        print(f"  Scheduler: {scheduler_display}")
        print(f"  Balance Weight: {args.balance_weight}")
        print(f"  Num Workers: {args.num_workers}")
        print(f"  GPUs: {world_size}")
        print(f"  Model: {args.model}")
        if model_layers_cfg is not None:
            if args.k_config is not None:
                print(f"  MoCE k-config (CLI): {args.k_config}")
            elif args.k is not None:
                print(f"  MoCE k (global CLI): {args.k}")
            else:
                print(f"  MoCE k: model defaults from models_unified.py")
        print(f"  AMP: {args.amp}")
        if args.dataset == 'cifar100':
            print(f"  Data Root: {args.cifar_root}")
        else:
            print(f"  Data Root: {args.imagenet_root}")
        print(f"{'='*80}\n")
    
    # Get data loaders once (shared across all models)
    batch_size_per_gpu = args.batch_size // world_size
    trainloader, testloader, train_sampler = get_dataloader(
        batch_size_per_gpu, rank, world_size, args
    )
    
    # Store results
    all_results = []
    num_classes = dataset_num_classes(args.dataset)
    
    # if args.model == 'baseline':
    #     model = resnet50_baseline(num_classes=num_classes, dataset=args.dataset)
    #     model_name = "ResNet-50 Baseline"
    # else:
    #     model = resnet50_moce_projection(
    #         sampling_factor=4,
    #         num_classes=num_classes,
    #         dataset=args.dataset
    #     )
    #     model_name = "ResNet-50 + MOCE"

    # ------------------------------------------------------------
    # Model Selection
    # ------------------------------------------------------------

    if args.model == 'baseline50':
        model = resnet50_baseline(num_classes=num_classes, dataset=args.dataset)
        model_name = "ResNet-50 Baseline"

    elif args.model == 'baseline101':
        model = resnet101_baseline(num_classes=num_classes, dataset=args.dataset)
        model_name = "ResNet-101 Baseline"

    elif args.model == 'baseline152':
        model = resnet152_baseline(num_classes=num_classes, dataset=args.dataset)
        model_name = "ResNet-152 Baseline"

    elif args.model == 'moce50':
        moce_kwargs = dict(
            sampling_factor=4,
            num_classes=num_classes,
            dataset=args.dataset,
        )
        if moce_k is not None:
            moce_kwargs["k"] = moce_k
        model = resnet50_moce_projection(**moce_kwargs)
        model_name = "ResNet-50 + MOCE"

    elif args.model == 'moce101':
        moce_kwargs = dict(
            sampling_factor=4,
            num_classes=num_classes,
            dataset=args.dataset,
        )
        if moce_k is not None:
            moce_kwargs["k"] = moce_k
        model = resnet101_moce_projection(**moce_kwargs)
        model_name = "ResNet-101 + MOCE"

    elif args.model == 'moce152':
        moce_kwargs = dict(
            sampling_factor=4,
            num_classes=num_classes,
            dataset=args.dataset,
        )
        if moce_k is not None:
            moce_kwargs["k"] = moce_k
        model = resnet152_moce_projection(**moce_kwargs)
        model_name = "ResNet-152 + MOCE"

    else:
        raise ValueError(f"Unsupported model {args.model}")
    
    model = model.to(device)


    # if rank == 0:
    #     flops, params = compute_model_flops_params(model, device, args, num_classes)
    #     print(f"FLOPs (1x forward): {flops/1e9:.3f} GFLOPs")
    #     print(f"Params: {params/1e6:.3f} M")

    # model = torch.compile(model, mode="max-autotune")
    # print(f"Model compiled with torch.compile (mode=max-autotune).....................................")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    results = train_model(model, model_name, trainloader, testloader,
                         train_sampler, device, world_size, rank, args,run_dir, last_ckpt_path, best_ckpt_path)
    all_results.append(results)
    
    del model
    torch.cuda.empty_cache()
    
    # ========================================================================
    # Final Comparison
    # ========================================================================
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"FINAL COMPARISON - ALL MODELS")
        print(f"{'='*80}\n")
        
        # Table header
        print(f"{'Model':<40} | {'Params':<12} | {'Best Acc':<10} | {'Time/Epoch':<10}")
        print(f"{'-'*40}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")
        
        # Sort by best accuracy (descending)
        sorted_results = sorted(all_results, key=lambda x: x['best_acc'], reverse=True)
        
        for r in sorted_results:
            print(f"{r['model_name']:<40} | {r['parameters']:>10,}  | "
                  f"{r['best_acc']:>8.2f}%  | "
                  f"{r['time_per_epoch']:>8.1f}s")
        
        print(f"\n{'='*80}")
        print(f"Testing Complete!")
        print(f"{'='*80}\n")
        
        # Highlight best performers
        best_acc_model = max(all_results, key=lambda x: x['best_acc'])
        fastest_model = min(all_results, key=lambda x: x['time_per_epoch'])
        
        print("Key Findings:")
        print(f"   Best Accuracy: {best_acc_model['model_name']} ({best_acc_model['best_acc']:.2f}%)")
        print(f"   Fastest: {fastest_model['model_name']} ({fastest_model['time_per_epoch']:.1f}s/epoch)")
        print()
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
