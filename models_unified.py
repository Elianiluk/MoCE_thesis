"""
Unified ResNet-50 implementation.
Includes:
1. Baseline ResNet-50
2. ResNet-50 + MoCE projection
Supports CIFAR-100 and ImageNet stems.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from MoCE import MoCE





# ------------------------------------------------------------------
# Per-block k configuration
# Edit these at the top of the file.
# Format: one list per stage, one value per block in that stage.
# ------------------------------------------------------------------
DEFAULT_MOCE_K_50 = [
    [8, 8, 8],                   # layer1
    [12, 12, 12, 12],                # layer2
    [24, 24, 24, 24, 24, 24],          # layer3
    [48, 48, 48],                   # layer4
]

DEFAULT_MOCE_K_101 = [
    [8, 8, 8],                   # layer1
    [8, 8, 8, 8],                # layer2
    [8] * 23,                    # layer3
    [8, 8, 8],                   # layer4
]

DEFAULT_MOCE_K_152 = [
    [8, 8, 8],                   # layer1
    [8] * 8,                     # layer2
    [8] * 36,                    # layer3
    [8, 8, 8],                   # layer4
]


def _normalize_k_config(k, layers_config):
    """
    Accept:
      - int -> same k for all blocks
      - nested list -> k per block per stage
    """
    if isinstance(k, int):
        return [[k for _ in range(num_blocks)] for num_blocks in layers_config]

    if not isinstance(k, (list, tuple)) or len(k) != len(layers_config):
        raise ValueError(
            f"k must be an int or a list with {len(layers_config)} stage lists."
        )

    normalized = []
    for stage_idx, (stage_k, num_blocks) in enumerate(zip(k, layers_config)):
        if not isinstance(stage_k, (list, tuple)) or len(stage_k) != num_blocks:
            raise ValueError(
                f"k[{stage_idx}] must contain exactly {num_blocks} values."
            )
        normalized.append(list(stage_k))

    return normalized


class Bottleneck(nn.Module):
    """Standard ResNet-50 bottleneck block."""
    
    expansion = 4
    
    def __init__(self, in_channels, mid_channels, stride=1):
        super(Bottleneck, self).__init__()
        
        # self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.conv1 = Conv1x1(in_channels, mid_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, 1, bias=False)
        self.conv3 = Conv1x1(mid_channels, mid_channels * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != mid_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(mid_channels * self.expansion)
            )
        
        self.relu3 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out



class BottleneckWithMoCEProjection(nn.Module):
    """
    ResNet-50 bottleneck with MoCE (with projection support).
    Uses fixed sampling factor and adds projection when needed.
    """
    
    expansion = 4
    
    def __init__(self, in_channels, mid_channels, stride=1, sampling_factor=4, k=8):
        super(BottleneckWithMoCEProjection, self).__init__()
        
        # MoCE replaces the first 1x1 conv (squeezing)
        self.moce = MoCE(in_channels, sampling_factor=sampling_factor, k=k)
        
        # Compute expected MoCE output channels
        squeezed_channels = in_channels // sampling_factor
        
        # Add projection if needed
        if squeezed_channels != mid_channels:
            self.projection = nn.Conv2d(squeezed_channels, mid_channels, 1, bias=False)
            self.bn_proj = nn.BatchNorm2d(mid_channels)
        else:
            self.projection = None
            self.bn1 = nn.BatchNorm2d(squeezed_channels)
        
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != mid_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(mid_channels * self.expansion)
            )
        
        self.relu3 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # MoCE forward
        out = self.moce(x)
        
        # Apply projection if needed
        if self.projection is not None:
            out = self.bn_proj(self.projection(out))
        else:
            out = self.bn1(out)

        # Rest of bottleneck
        out = self.relu1(out)
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet50(nn.Module):
    """ResNet-50 with dataset-specific input stem."""
    
    def __init__(self, block, num_classes=100, reduction_ratios=None, sampling_factor=None,
                 dataset='cifar100', k=8, layers_config=None):
        super(ResNet50, self).__init__()
        
        self.in_channels = 64
        self.block = block
        self.dataset = dataset.lower()
        self.k = k
        self.layers_config = layers_config or [3, 4, 6, 3]
        if self.dataset not in {'cifar100', 'imagenet'}:
            raise ValueError(f"Unsupported dataset '{dataset}'. Use 'cifar100' or 'imagenet'.")
        
        if block == BottleneckWithMoCEProjection:
            if sampling_factor is None:
                sampling_factor = 4
            self.sampling_factor = sampling_factor
            self.reduction_ratios = None
            self.k_per_block = _normalize_k_config(k, self.layers_config)
        else:
            self.reduction_ratios = None
            self.sampling_factor = None
            self.k_per_block = None
        
        # Dataset-specific stem.
        if self.dataset == 'imagenet':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, self.layers_config[0], stride=1, layer_idx=0)
        self.layer2 = self._make_layer(block, 128, self.layers_config[1], stride=2, layer_idx=1)
        self.layer3 = self._make_layer(block, 256, self.layers_config[2], stride=2, layer_idx=2)
        self.layer4 = self._make_layer(block, 512, self.layers_config[3], stride=2, layer_idx=3)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self._initialize_weights()

    def _make_layer(self, block, mid_channels, blocks, stride, layer_idx):
        layers = []

        # -------- First block in stage --------
        if block == Bottleneck:
            layers.append(Bottleneck(self.in_channels, mid_channels, stride))

        elif block == BottleneckWithMoCEProjection:
            if layer_idx == 0:
                sf = 1
            else:
                sf = 2

            layers.append(
                BottleneckWithMoCEProjection(
                    self.in_channels,
                    mid_channels,
                    stride,
                    sampling_factor=sf,
                    k=self.k_per_block[layer_idx][0],
                )
            )

        self.in_channels = mid_channels * block.expansion

        # -------- Remaining blocks --------
        for block_idx in range(1, blocks):
            if block == Bottleneck:
                layers.append(Bottleneck(self.in_channels, mid_channels, 1))

            elif block == BottleneckWithMoCEProjection:
                layers.append(
                    BottleneckWithMoCEProjection(
                        self.in_channels,
                        mid_channels,
                        1,
                        sampling_factor=4,
                        k=self.k_per_block[layer_idx][block_idx],
                    )
                )

        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        # from MoCE import MoCE
        for m in self.modules():
            if isinstance(m, MoCE):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def resnet50_baseline(**kwargs):
    """ResNet-50 Baseline (no MoCE)."""
    return ResNet50(Bottleneck, layers_config=[3, 4, 6, 3], **kwargs)



def resnet50_moce_projection(sampling_factor=4, k=DEFAULT_MOCE_K_50, **kwargs):
    """
    ResNet-50 + MoCE (with projection support).
    Uses fixed sampling factor (2, 4, or 8).
    """
    return ResNet50(
        BottleneckWithMoCEProjection,
        sampling_factor=sampling_factor,
        k=k,
        layers_config=[3, 4, 6, 3],
        **kwargs
    )

def _replace_layers(model, layers_config):
    model.layers_config = layers_config
    model.in_channels = 64

    # recompute k configuration
    if model.block == BottleneckWithMoCEProjection:
        model.k_per_block = _normalize_k_config(model.k, model.layers_config)

    model.layer1 = model._make_layer(model.block, 64,  layers_config[0], stride=1, layer_idx=0)
    model.layer2 = model._make_layer(model.block, 128, layers_config[1], stride=2, layer_idx=1)
    model.layer3 = model._make_layer(model.block, 256, layers_config[2], stride=2, layer_idx=2)
    model.layer4 = model._make_layer(model.block, 512, layers_config[3], stride=2, layer_idx=3)

    return model


# ---------------------------
# Baseline Variants
# ---------------------------

def resnet101_baseline(**kwargs):
    model = ResNet50(Bottleneck, layers_config=[3, 4, 23, 3], **kwargs)
    return _replace_layers(model, [3, 4, 23, 3])


def resnet152_baseline(**kwargs):
    model = ResNet50(Bottleneck, layers_config=[3, 8, 36, 3], **kwargs)
    return _replace_layers(model, [3, 8, 36, 3])


# ---------------------------
# MoCE Projection Variants
# ---------------------------

def resnet101_moce_projection(sampling_factor=4, k=DEFAULT_MOCE_K_101, **kwargs):
    model = ResNet50(
        BottleneckWithMoCEProjection,
        sampling_factor=sampling_factor,
        k=k,
        layers_config=[3, 4, 23, 3],
        **kwargs
    )
    return _replace_layers(model, [3, 4, 23, 3])


def resnet152_moce_projection(sampling_factor=4, k=DEFAULT_MOCE_K_152, **kwargs):
    model = ResNet50(
        BottleneckWithMoCEProjection,
        sampling_factor=sampling_factor,
        k=k,
        layers_config=[3, 8, 36, 3],
        **kwargs
    )
    return _replace_layers(model, [3, 8, 36, 3])