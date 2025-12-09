# resnet_conformer_model.py (GroupNorm Version for Small Batch)
# ResNet (GN) + Projection + Conformer + Stable Swish

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools

# ----------------- Loss Functions (No Change) -----------------
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary focal loss (for multi-label SED).
    inputs:  logits, shape [..., C]
    targets: {0,1}, shape [..., C]
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCE with optional pos_weight
        pos_w = None
        if self.pos_weight is not None:
            pos_w = self.pos_weight.to(inputs.device).type_as(inputs)

        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=pos_w
        )  # [..., C]

        # pt = P(y_hat == y)
        p = torch.sigmoid(inputs)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)

        alpha = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
        alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)

        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class SeldLoss(nn.Module):
    """
    Simplified multi-ACCDOA loss (no ADPIT permutations).
    - Uses only the first K tracks from the ADPIT labels.
    - target: [B, T, 6, 5, C]  (tracks, [act,x,y,z,dist], classes)
    - output:
        'sed' : [B, T, C*K]       (logits)
        'doa' : [B, T, C*K*3]     (tanh-ed xyz)
        'dist': [B, T, C*K]       (ReLU-ed distance)
    """
    def __init__(self, params):
        super().__init__()
        self.nb_classes = int(params["unique_classes"])
        self.nb_tracks  = 3  # number of output tracks

        # ---- Loss weights ----
        self.sed_weight  = float(params.get("sed_weight", 4.0))
        self.doa_weight  = float(params.get("doa_weight", 1.0))
        self.dist_weight = float(params.get("dist_weight", 0.3))

        # ---- Distance scaling ----
        self.dist_scale  = float(params.get("dist_scale", 10.0))

        # ---- SED loss (focal or BCE) ----
        use_focal       = bool(params.get("use_focal_for_sed", False))
        focal_gamma     = float(params.get("focal_gamma", 2.0))
        # 클래스 불균형 심하긴 하지만, 일단 pos_weight는 끄고 시작
        # (과검출/과소검출 패턴부터 바로잡는 게 우선)
        # sed_pos_weight  = float(params.get("sed_pos_weight", 1.0))

        # ---- SED: Plain BCEWithLogitsLoss (no pos_weight) ----
        if use_focal:
            self.sed_loss_fn = FocalLoss(
                gamma=focal_gamma,
                pos_weight=None,
                reduction="none",
            )
        else:
            self.sed_loss_fn = nn.BCEWithLogitsLoss(
                reduction="none",   # pos_weight=None
            )


        # ---- DOA / Distance loss ----
        self.doa_loss_fn  = nn.MSELoss(reduction="none")
        self.dist_loss_fn = nn.SmoothL1Loss(beta=0.5, reduction="none")

    def forward(self, output, target):
        # target: [B, T, 6, 5, C]
        if target.ndim != 5:
            raise RuntimeError(f"[SeldLoss] Expected target [B,T,6,5,C], got {target.shape}")
        B, T, L, A, C = target.shape

        if L < self.nb_tracks:
            raise RuntimeError(f"[SeldLoss] target has fewer tracks ({L}) than nb_tracks ({self.nb_tracks})")

        if A < 5:
            raise RuntimeError(f"[SeldLoss] expected 5 axes (act,xyz,dist), got {A}")

        if C != self.nb_classes:
            # 라벨에 패딩 클래스가 있을 경우 잘라냄
            target = target[..., :self.nb_classes]
            C = self.nb_classes

        K = self.nb_tracks

        # ---- 예측값 reshape ----
        sed_pred  = output["sed"]   # [B,T,C*K]
        doa_pred  = output["doa"]   # [B,T,C*K*3]
        dist_pred = output["dist"]  # [B,T,C*K]

        if sed_pred.shape[0] != B or sed_pred.shape[1] != T:
            raise RuntimeError(
                f"[SeldLoss] Mismatch between output time dim {sed_pred.shape[:2]} and target {(B,T)}"
            )

        sed_pred  = sed_pred.view(B, T, K, C)       # [B,T,K,C]
        doa_pred  = doa_pred.view(B, T, K, C, 3)    # [B,T,K,C,3]
        dist_pred = dist_pred.view(B, T, K, C)      # [B,T,K,C]

        # ---- ADPIT label에서 앞 K 트랙만 사용 ----
        # target_k: [B,T,K,5,C]
        target_k = target[:, :, :K, :, :].contiguous()

        act_gt   = target_k[:, :, :, 0, :]                     # [B,T,K,C]
        xyz_gt   = target_k[:, :, :, 1:4, :].permute(0,1,2,4,3) # [B,T,K,C,3]
        dist_gt  = target_k[:, :, :, 4, :]                     # [B,T,K,C]

        device = sed_pred.device

        # ---- SED loss ----
        sed_loss = self.sed_loss_fn(sed_pred, act_gt.to(device)).mean()

        # ---- DOA / DIST loss (active 위치에서만) ----
        active_mask = (act_gt > 0.5).to(device)   # [B,T,K,C]

        if active_mask.any():
            # DOA: xyz 차이
            doa_diff = self.doa_loss_fn(doa_pred, xyz_gt.to(device))  # [B,T,K,C,3]
            doa_loss = doa_diff[active_mask.unsqueeze(-1).expand_as(doa_diff)].mean()

            # Dist: SmoothL1, 거리 스케일 정규화
            dist_diff = self.dist_loss_fn(
                dist_pred / self.dist_scale,
                dist_gt.to(device) / self.dist_scale,
            )  # [B,T,K,C]
            dist_loss = dist_diff[active_mask].mean()
        else:
            doa_loss  = sed_loss.new_tensor(0.0)
            dist_loss = sed_loss.new_tensor(0.0)

        total = (
            self.sed_weight  * sed_loss
            + self.doa_weight  * doa_loss
            + self.dist_weight * dist_loss
        )
        return total



# ----------------- Model Components (GN applied) -----------------

class Swish(nn.Module):
    def forward(self, x):
        # [★ FIX] Force FP32 for Swish
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            return x * torch.sigmoid(x)

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, x): return F.glu(x, dim=self.dim)

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
    def forward(self, x): return self.conv(x)

class ConformerConvModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1, bias=True)
        self.glu = GLU(dim=1)
        pad = (kernel_size - 1) // 2
        self.depthwise_conv = DepthwiseConv1d(dim, dim, kernel_size, padding=pad, bias=True)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.swish = Swish()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.layer_norm(x).transpose(1, 2)
        x = self.conv1(x); x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x); x = self.swish(x)
        x = self.conv2(x); x = self.dropout(x).transpose(1, 2)
        return x + residual

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * mult), Swish(), 
                                 nn.Dropout(dropout), nn.Linear(dim * mult, dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class ConformerBlock(nn.Module):
    def __init__(self, dim, n_heads=8, attn_dropout=0.1, ff_dropout=0.1, conv_dropout=0.1, conv_kernel_size=31):
        super().__init__()
        self.ff1  = FeedForward(dim, dropout=ff_dropout)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=attn_dropout, batch_first=True)
        self.conv = ConformerConvModule(dim, kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2  = FeedForward(dim, dropout=ff_dropout)
        self.norm1= nn.LayerNorm(dim)
        self.norm2= nn.LayerNorm(dim)

    def forward(self, x):
        x = x.clamp_(-1e3, 1e3)
        if not x.is_contiguous(): x = x.contiguous()
        
        # FFN1
        x = x + 0.5 * self.ff1(x)
        
        # MHA
        qkv = x
        with torch.cuda.amp.autocast(enabled=False):
            q_in = qkv.float().contiguous()
            attn_out, _ = self.attn(q_in, q_in, q_in)
        x = x + attn_out.to(x.dtype)
        
        # Conv
        x = x + self.conv(x)
        
        # FFN2
        x = x + 0.5 * self.ff2(x)
        x = self.norm2(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# [★수정] BasicBlock에서 BatchNorm 대신 GroupNorm 사용
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes) <-- 삭제
        self.bn1 = nn.GroupNorm(num_groups=32, num_channels=planes) # <-- GroupNorm 추가
        
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes) <-- 삭제
        self.bn2 = nn.GroupNorm(num_groups=32, num_channels=planes) # <-- GroupNorm 추가
        
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None: residual = self.downsample(x)
        out += residual
        return self.relu(out)

# ----------------- Main Model -----------------

class SeldModel(nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.params = params
        self.nb_classes = params['unique_classes']
        self.in_channels = in_feat_shape[1]
        self.dist_scale = params.get('dist_scale', 50.0)

        # ResNet Backbone (First Layer BN -> GN)
        self.resnet = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            # nn.BatchNorm2d(64), <-- 삭제
            nn.GroupNorm(num_groups=32, num_channels=64), # <-- GroupNorm 교체
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.in_channels = 64
        self.layer1 = self._make_layer(BasicBlock, 64, params['resnet_layers'][0])
        self.layer2 = self._make_layer(BasicBlock, 128, params['resnet_layers'][1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, params['resnet_layers'][2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, params['resnet_layers'][3], stride=2)

        # Dimension Calculation
        F_in = in_feat_shape[3]
        self._expected_Fp = math.ceil(F_in / 32)
        resnet_out_dim = 512 * self._expected_Fp 

        # Projection (Memory Optimization)
        self.conformer_dim = 512 
        self.projection = nn.Linear(resnet_out_dim, self.conformer_dim)
        
        # Conformer Encoder
        self.conformer_encoder = nn.Sequential(
            *[ConformerBlock(
                dim=self.conformer_dim, n_heads=8,
                ff_dropout=params['dropout_rate'], attn_dropout=params['dropout_rate'],
                conv_dropout=params['dropout_rate'], conv_kernel_size=params['conformer_conv_kernel_size']
            ) for _ in range(params['conformer_n_layers'])]
        )
        self.upsample = nn.Upsample(size=params['label_sequence_length'], mode='linear', align_corners=False)

        # Heads
        prior = float(self.params.get('sed_prior', 0.05))
        prior = min(max(prior, 1e-4), 1 - 1e-4)
        bias  = math.log(prior / (1.0 - prior))

        self.nb_tracks = 3
        out_sed  = self.nb_classes * self.nb_tracks              # [C * K]
        out_doa  = self.nb_classes * 3 * self.nb_tracks          # [C * K * 3]
        out_dist = self.nb_classes * self.nb_tracks              # [C * K]

        hidden = self.conformer_dim  # 512

        # ---- NERC-style SED head: FC-LReLU-FC-LReLU-FC ----
        self.sed_head = nn.Sequential(
            nn.Linear(self.conformer_dim, hidden),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(hidden, out_sed),
        )
        # prior 기반 bias 초기화는 마지막 layer에 그대로 적용
        nn.init.constant_(self.sed_head[-1].bias, bias)

        # ---- NERC-style DOA head ----
        self.doa_head = nn.Sequential(
            nn.Linear(self.conformer_dim, hidden),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(hidden, out_doa),
        )

        # ---- NERC-style Distance(SDE) head ----
        self.dist_head = nn.Sequential(
            nn.Linear(self.conformer_dim, hidden),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(hidden, out_dist),
        )


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion) <-- 삭제
                nn.GroupNorm(num_groups=32, num_channels=planes * block.expansion) # <-- 교체
            )
        layers = [block(self.in_channels, planes, stride, downsample)]
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, Cin, Tin, Fin]
        if not torch.isfinite(x).all(): x = torch.nan_to_num(x, 0.0)

        x = self.resnet(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        B, C, Tp, Fp = x.shape 

        # [B, Tp, 2048] -> [B, Tp, 512] 
        x = x.permute(0,2,1,3).contiguous().view(B, Tp, C*Fp)
        x = self.projection(x) 

        x = self.conformer_encoder(x)
        # x: [B, Tp, D]

        # label_sequence_length로 upsample (이 부분은 그대로 유지)
        x = x.transpose(1, 2)                # [B, D, Tp]
        x = self.upsample(x)                 # [B, D, T_label]
        x = x.transpose(1, 2)                # [B, T_label, D]

        # ---- NERC-style heads ----
        sed_logits = self.sed_head(x)        # [B, T, C*K]
        doa_raw    = self.doa_head(x)        # [B, T, C*K*3]
        dist_raw   = self.dist_head(x)       # [B, T, C*K]

        # 활성함수는 리포트와 동일하게
        sed_output  = sed_logits             # 로스에서 BCEWithLogits 쓰므로 그대로 logits
        doa_output  = torch.tanh(doa_raw)    # [-1, 1] 범위
        dist_output = F.relu(dist_raw)       # 거리 >= 0

        return {"sed": sed_output, "doa": doa_output, "dist": dist_output}
