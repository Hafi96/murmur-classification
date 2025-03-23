import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Hierachical_MS_Net(nn.Module):
    def __init__(self, 
                block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock,
                stagewise_layers: List[int] = [2, 2, 2, 2],
                scalewise_inplanes: List[int] = [32, 16, 16],
                include_patient_data: bool = False,
                num_classes: int = 2,  # Updated to exclude "Unknown"
                zero_init_residual: bool = True,
                ):
        super().__init__()
        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        
        self.scalewise_inplanes = scalewise_inplanes
        self.conv1_scale1 = nn.Conv2d(1, self.scalewise_inplanes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_scale1 = self._norm_layer(self.scalewise_inplanes[0])
        self.conv1_scale2 = nn.Conv2d(1, self.scalewise_inplanes[1], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_scale2 = self._norm_layer(self.scalewise_inplanes[1])
        self.conv1_scale3 = nn.Conv2d(1, self.scalewise_inplanes[2], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_scale3 = self._norm_layer(self.scalewise_inplanes[2])
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1_scale1 = self._make_layer(block, self.scalewise_inplanes[0], 64, stagewise_layers[0])
        self.layer2_scale1 = self._make_layer(block, 64, 128, stagewise_layers[0], stride=2)
        self.layer1_scale2 = self._make_layer(block, self.scalewise_inplanes[1], 32, stagewise_layers[0])
        
        self.layer1_scale12 = self._make_layer(block, 128 + 32, 256, stagewise_layers[1])
        self.layer2_scale12 = self._make_layer(block, 256, 256, stagewise_layers[1], stride=2)
        self.layer1_scale3 = self._make_layer(block, self.scalewise_inplanes[2], 32, stagewise_layers[1])
        
        self.layer1_scale123 = self._make_layer(block, 256 + 32, 512, stagewise_layers[2], stride=2)
        self.layer2_scale123 = self._make_layer(block, 512, 512, stagewise_layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.include_patient_data = include_patient_data
        
        if include_patient_data:
            ch = 256
            self.mlp_patient_features = nn.Sequential(
                nn.Linear(6, ch),
                nn.ReLU(True),
                nn.Linear(ch, ch),
                nn.ReLU(True),
                nn.Linear(ch, ch),
                nn.ReLU(True),
                nn.Linear(ch, ch),
                nn.ReLU(True),
            )
            self.fc = nn.Linear(512 * block.expansion + ch, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        inplanes: int, 
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(inplanes, planes, stride, downsample, self.groups, self.base_width, norm_layer)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, groups=self.groups, base_width=self.base_width, norm_layer=norm_layer))
        return nn.Sequential(*layers)
        
    def forward(self, multi_scale_spectrograms, patient_features=None):
        spectrogram_scale1, spectrogram_scale2, spectrogram_scale3 = multi_scale_spectrograms
        
        s1 = self.relu(self.bn_scale1(self.conv1_scale1(spectrogram_scale1)))
        s2 = self.relu(self.bn_scale2(self.conv1_scale2(spectrogram_scale2)))
        s3 = self.relu(self.bn_scale3(self.conv1_scale3(spectrogram_scale3)))
        
        h1_s1 = self.layer1_scale1(s1)
        h1_s1 = self.layer2_scale1(h1_s1)
        h1_s2 = self.layer1_scale2(s2)
        h1_s12 = torch.cat((h1_s1, h1_s2), dim=1)
        
        h2_s12 = self.layer1_scale12(h1_s12)
        h2_s12 = self.layer2_scale12(h2_s12)
        h2_s3 = self.layer1_scale3(s3)
        h2_s123 = torch.cat((h2_s12, h2_s3), dim=1)
        
        h3_s123 = self.layer1_scale123(h2_s123)
        h4_s123 = self.layer2_scale123(h3_s123)
        
        h = self.avgpool(h4_s123)
        h = torch.flatten(h, 1)
        
        if self.include_patient_data:
            h_patient_features = self.mlp_patient_features(patient_features)
            h = torch.cat((h, h_patient_features), dim=1)
            
        out = self.fc(h)

        return out
