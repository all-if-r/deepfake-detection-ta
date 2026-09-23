"""
Arsitektur Model EfficientNet-B4 + CBAM + Non-Local Block (Tugas 4 & 5)
Sesuai Proposal & Subbab 4.1.1:
1. Backbone: EfficientNet-B4 pre-trained IMAGENET1K (input 380x380x3)
2. Freeze stage 1-4, fine-tune stage 5-7
3. CBAM (Channel & Spatial Attention, reduction ratio r=16) disisipkan pada stage 5, 6, 7
4. Non-Local Block (Embedded Gaussian) disisipkan 1x di akhir stage 7 (setelah CBAM stage 7, sebelum GAP)
5. Head Klasifikasi: GAP -> Dropout (0.5) -> Linear (2 neuron) -> Softmax
6. Threshold biner: Fake jika P(Fake) >= 0.5
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, List


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) pada CBAM.
    M_c(F) = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) pada CBAM.
    M_s(F) = σ(f^{7x7}([AvgPool(F); MaxPool(F)]))
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size harus 3 atau 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return x * scale


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Menggabungkan Channel Attention dan Spatial Attention secara sekuensial.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class NonLocalBlock(nn.Module):
    """
    Non-Local Block (Varian Embedded Gaussian).
    y = Softmax(θ(x) @ φ(x)^T / sqrt(d_k)) @ g(x)
    z = W_z(y) + x
    """
    def __init__(self, in_channels: int, inter_channels: int = None):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else max(1, in_channels // 2)

        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W_z = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )
        # Inisialisasi bobot W_z ke nol agar blok awalnya bersifat identity mapping
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # g(x) -> (B, inter_c, H*W) -> transpose to (B, H*W, inter_c)
        g_x = self.g(x).view(b, self.inter_channels, -1).permute(0, 2, 1)

        # theta(x) -> (B, H*W, inter_c)
        theta_x = self.theta(x).view(b, self.inter_channels, -1).permute(0, 2, 1)

        # phi(x) -> (B, inter_c, H*W)
        phi_x = self.phi(x).view(b, self.inter_channels, -1)

        # Energy matrix f = theta @ phi -> (B, H*W, H*W)
        energy = torch.matmul(theta_x, phi_x)
        # Scaling factor 1/sqrt(inter_channels) untuk stabilitas Softmax
        energy = energy / (self.inter_channels ** 0.5)
        attn = torch.softmax(energy, dim=-1)

        # Matmul attn @ g_x -> (B, H*W, inter_c) -> permute back to (B, inter_c, H, W)
        y = torch.matmul(attn, g_x).permute(0, 2, 1).contiguous().view(b, self.inter_channels, h, w)
        
        # Residual connection
        z = self.W_z(y) + x
        return z


class EfficientNetB4_CBAM_NonLocal(nn.Module):
    """
    Model Lengkap Konfigurasi K5:
    - Backbone EfficientNet-B4
    - Stage 1-4 Frozen
    - Stage 5-7 Fine-Tuned + CBAM di Stage 5, 6, 7
    - Non-Local Block di akhir Stage 7 (sebelum GAP)
    - Head: GAP -> Dropout (0.5) -> Linear (2) -> Softmax
    """
    def __init__(
        self, 
        pretrained: bool = True, 
        num_classes: int = 2, 
        dropout_rate: float = 0.5,
        cbam_reduction: int = 16,
        use_cbam: bool = True,
        use_non_local: bool = True
    ):
        super().__init__()
        self.use_cbam = use_cbam
        self.use_non_local = use_non_local
        
        # Inisialisasi backbone EfficientNet-B4
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.efficientnet_b4(weights=weights)

        # Segmentasi stage backbone
        # features[0] = Stem Conv
        # features[1] = Stage 1 (MBConv1) - in 32 -> out 24
        # features[2] = Stage 2 (MBConv6) - in 24 -> out 32
        # features[3] = Stage 3 (MBConv6) - in 32 -> out 56
        # features[4] = Stage 4 (MBConv6) - in 56 -> out 112
        # features[5] = Stage 5 (MBConv6) - in 112 -> out 160
        # features[6] = Stage 6 (MBConv6) - in 160 -> out 272
        # features[7] = Stage 7 (MBConv6) - in 272 -> out 448
        # features[8] = Head Conv - in 448 -> out 1792

        self.stem = base_model.features[0]
        self.stage1 = base_model.features[1]
        self.stage2 = base_model.features[2]
        self.stage3 = base_model.features[3]
        self.stage4 = base_model.features[4]
        
        self.stage5 = base_model.features[5]
        self.stage6 = base_model.features[6]
        self.stage7 = base_model.features[7]

        # CBAM modules (kondisional berdasarkan flag use_cbam)
        if self.use_cbam:
            self.cbam5 = CBAM(in_channels=160, reduction_ratio=cbam_reduction)
            self.cbam6 = CBAM(in_channels=272, reduction_ratio=cbam_reduction)
            self.cbam7 = CBAM(in_channels=448, reduction_ratio=cbam_reduction)
        else:
            self.cbam5 = None
            self.cbam6 = None
            self.cbam7 = None
        
        # Non-Local Block (kondisional berdasarkan flag use_non_local)
        if self.use_non_local:
            self.non_local = NonLocalBlock(in_channels=448)
        else:
            self.non_local = None

        # Head Conv expansion (1792 channels)
        self.head_conv = base_model.features[8]

        # Classification Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(1792, num_classes)

        # Terapkan pembekuan bobot (Freeze Stage 1-4)
        self._freeze_stages_1_to_4()

    def _freeze_stages_1_to_4(self):
        """
        Membekukan gradien pada Stem dan Stage 1 s.d. 4.
        """
        frozen_modules = [self.stem, self.stage1, self.stage2, self.stage3, self.stage4]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen stages
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Trainable stages + CBAM & Non-Local (kondisional)
        x = self.stage5(x)
        if self.use_cbam and self.cbam5 is not None:
            x = self.cbam5(x)

        x = self.stage6(x)
        if self.use_cbam and self.cbam6 is not None:
            x = self.cbam6(x)

        x = self.stage7(x)
        if self.use_cbam and self.cbam7 is not None:
            x = self.cbam7(x)

        if self.use_non_local and self.non_local is not None:
            x = self.non_local(x)

        x = self.head_conv(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        pooled = self.gap(feat)
        flat = torch.flatten(pooled, 1)
        dropped = self.dropout(flat)
        logits = self.fc(dropped)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mengembalikan probabilitas Softmax [P(Real), P(Fake)].
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_parameter_groups(self, lr_backbone: float = 1e-5, lr_head: float = 1e-4) -> List[dict]:
        """
        Memisahkan parameter untuk differential learning rate:
        - Group 1: Stage 5-7 + CBAMs (jika aktif) + Non-Local (jika aktif) + Head Conv (lr = lr_backbone)
        - Group 2: Classifier Head FC (lr = lr_head)
        """
        backbone_trainable_params = []
        trainable_modules = [self.stage5, self.stage6, self.stage7]
        if self.use_cbam and self.cbam5 is not None:
            trainable_modules.extend([self.cbam5, self.cbam6, self.cbam7])
        if self.use_non_local and self.non_local is not None:
            trainable_modules.append(self.non_local)
        trainable_modules.append(self.head_conv)

        for mod in trainable_modules:
            for param in mod.parameters():
                if param.requires_grad:
                    backbone_trainable_params.append(param)

        head_params = list(self.fc.parameters())

        return [
            {"params": backbone_trainable_params, "lr": lr_backbone, "name": "backbone_stage5_7"},
            {"params": head_params, "lr": lr_head, "name": "classifier_head"}
        ]
