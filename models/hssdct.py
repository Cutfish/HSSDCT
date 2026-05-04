import torch,argparse
from torch import nn
import numpy as np, math
from torch.nn import functional as F
from torch.autograd import Variable
import functools
from .module_util import *
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
from ptflops import get_model_complexity_info

# ============================================================
# HSSDCT: Hierarchical Spatial-Spectral Dense Correlation Network
# Paper: "HSSDCT: Factorized Spatial-Spectral Correlation for
#        Hyperspectral Image Fusion" (arXiv:2602.00490)
#
# 本文件实现了HSSDCT网络的所有核心模块，包括：
#   1. DFE (Dual Feature Extraction) / SSFE - 空间-光谱特征提取模块 (论文 Section 2.3)
#   2. SCC (Spatial-Channel Correlation) / SSCL - 空间-光谱相关层 (论文 Section 2.3)
#   3. HierarchicalTransformerBlock - 分层Transformer块 (HDRTB的核心组件)
#   4. SwinBasedFeatFusionBlock - 分层密集残差Transformer块 HDRTB (论文 Section 2.2)
#   5. YDCFN - 双分支融合网络整体架构 (论文 Section 2.1 & Figure 1)
#   6. HyDCFN - 顶层包装器（含AWGN噪声注入）
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Convex-Optimization-Aware SR net')
    
    parser.add_argument('--SEED', type=int, default=1029)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--epochs', type=int, default=900)
    parser.add_argument('--lr_scheduler', type=str, default="cosine")
    parser.add_argument('--resume_ind', type=int, default=0)
    parser.add_argument('--resume_ckpt', type=str, default="")
    parser.add_argument('--snr', type=int, default=35)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=2)
    parser.add_argument('--finetuning_step', type=int, default=300, help='Works only if the mixed_align_opt is on')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay rate, 0 means training without weight decay')
    
    
    ## Data generator configuration
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--bands', type=int, default=172)
    parser.add_argument('--msi_bands', type=int, default=4)
    parser.add_argument('--mis_pix', type=int, default=0)
    parser.add_argument('--mixed_align_opt', type=int, default=0)
    parser.add_argument('--joint_loss', type=int, default=1)
    
    # Network architecture configuration
    parser.add_argument("--network_mode", type=int, default=1, help="Training network mode: 0) Single mode, 1) LRHSI+HRMSI, 2) COCNN (LRHSI+HRMSI+CO), Default: 2")     
    parser.add_argument('--num_base_chs', type=int, default=172, help='The number of the channels of the base feature')
    parser.add_argument('--num_blocks', type=int, default=6, help='The number of the repeated blocks in backbone')
    parser.add_argument('--num_agg_feat', type=int, default=172//4, help='the additive feature maps in the block')
    parser.add_argument('--groups', type=int, default=1, help="light version the group value can be >1, groups=1 for full COCNN version, groups=4 is COCNN-Light for 4 HRMSI version")
    
    # Others
    parser.add_argument("--root", type=str, default="/home/test/rdg/Fusion_data/", help='data root folder')   
    parser.add_argument("--val_file", type=str, default="./val.txt")   
    parser.add_argument("--train_file", type=str, default="./train.txt")   
    parser.add_argument("--prefix", type=str, default="DCSN_cocnn_light_adv")  
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:device_id or cpu")  
    parser.add_argument("--DEBUG", type=bool, default=False)  
    parser.add_argument("--gpus", type=int, default=1)  
    
    
    args = parser.parse_args()

    return args


class DFE(nn.Module):
    """ Dual Feature Extraction (DFE)
    
    【论文对应】SSFE (Spatial-Spectral Feature Extraction) 模块，见论文 Section 2.3 "Feature Projection" 及 Figure 4。
    SSFE采用双分支设计生成Query和Value表示：
      - 卷积路径（conv分支）：用于局部空间编码（local spatial encoding）
      - 线性路径（linear分支）：用于光谱编码/通道分割的矩阵分解（spectral encoding by channel splitting）
    两路特征通过逐元素相乘融合，产生同时富含空间和光谱上下文的Q和V表示，从而降低复杂度。

    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.
    """
    def __init__(self, in_features, out_features):
        super().__init__()

        self.out_features = out_features

        # [SSFE - 卷积分支] 用于局部空间编码：1x1降维 -> 3x3局部特征提取 -> 1x1升维
        # 对应论文 Figure 4 中 SSFE 的 Conv 分支（含 Conv 3x3 和 Linear/Conv 1x1 路径）
        self.conv = nn.Sequential(nn.Conv2d(in_features, in_features // 5, 1, 1, 0),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_features // 5, in_features // 5, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_features // 5, out_features, 1, 1, 0))
        
        # [SSFE - 线性分支] 用于光谱编码：通过1x1卷积实现通道间的线性投影
        # 对应论文 Figure 4 中 SSFE 的 Linear 分支
        self.linear = nn.Conv2d(in_features, out_features,1,1,0)

    def forward(self, x, x_size):
        """【DFE前向传播】双特征提取：生成富含空间+光谱信息的Q/V表示
        
        真实调用场景：
          光谱分支: x=[B, 16384, 160], x_size=(128,128), in_features=160, out_features=160
          空间分支: x=[B, 65536, 40],  x_size=(256,256), in_features=40,  out_features=40
        
        数据流（论文 Figure 4 / Section 2.3）:
          输入F(序列) → [转Conv2d格式] → conv(空间编码) * linear(光谱编码) → [转回序列] → Q/V输出
        """
        # 解包输入张量维度
        # 光谱分支: [B, 16384, 160];  空间分支: [B, 65536, 40]
        B, L, C = x.shape

        # 解包空间尺寸
        # 光谱分支: H=128, W=128;   空间分支: H=256, W=256
        H, W = x_size

        # ---- 序列格式 → CNN/Conv2d格式 ----
        # permute(0,2,1): 将[通道维,空间维]交换位置，从Transformer格式转为CNN格式
        #   光谱分支: [B, 16384, 160] → [B, 160, 16384]
        #   空间分支: [B, 65536, 40]  → [B, 40, 65536]
        #
        # contiguous(): permute只改stride不搬数据，内存变为non-contiguous
        #               view()要求物理连续存储，所以必须先拷贝重排
        #
        # view(B,C,H,W): 把1D空间维度L拆回2D(H×W)，得到标准4D Conv2d输入
        #   光谱分支: [B, 160, 16384] → [B, 160, 128, 128]
        #   空间分支: [B, 40, 65536]  → [B, 40, 256, 256]
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        # 光谱分支输出: [B, 160, 128, 128]
        # 空间分支输出: [B, 40, 256, 256]

        # ---- 双路径门控融合 (论文 Figure 4 的 SSFE 双分支设计) ----
        # ┌─────────────────── Conv 分支 (局部空间编码) ───────────────────┐
        # │                                                                 │
        # │  self.conv = Sequential(                                        │
        # │    Conv2d(C → C//5, k=1),     ← 1×1降维压缩                    │
        # │    LeakyReLU(0.2),                                             │
        # │    Conv2d(C//5 → C//5, k=3,p=1), ← 3×3提取局部空间纹理         │
        # │    LeakyReLU(0.2),                                             │
        # │    Conv2d(C//5 → C, k=1)      ← 1×1升维恢复                   │
        # │  )                                                              │
        # │                                                                 │
        # │  光谱分支: [B,160,128,128]                                      │
        # │    →Conv1x1(→32)→LReLU→[B,32,128,128]                         │
        # │    →Conv3x3(32→32)→LReLU→[B,32,128,128]  捕获3×3邻域空间细节   │
        # │    →Conv1x1(→160)→[B,160,128,128]                              │
        # │                                                                 │
        # │  空间分支: [B,40,256,256]                                       │
        # │    →Conv1x1(→8)→LReLU→[B,8,256,256]                           │
        # │    →Conv3x3(8→8)→LReLU→[B,8,256,256]                           │
        # │    →Conv1x1(→40)→[B,40,256,256]                                │
        # └─────────────────────────────────────────────────────────────────┘
        #                          *
        # ┌─────────────────── Linear 分支 (全局光谱编码) ─────────────────┐
        # │                                                                 │
        # │  self.linear = Conv2d(C → C, k=1)  即逐像素全连接              │
        # │                                                                 │
        # │  光谱分支: [B,160,128,128] → Conv1x1(160→160) → [B,160,128,128]│
        # │            (无激活函数，纯线性变换，建模通道间的全局关系)          │
        # │                                                                 │
        # │  空间分支: [B,40,256,256]  → Conv1x1(40→40)   → [B,40,256,256] │
        # └─────────────────────────────────────────────────────────────────┘
        #
        # * : 逐元素相乘 = 门控机制（类似GLU/SwiGLU）
        #     Linear分支的输出充当"软开关"，选择性通过Conv分支的空间特征
        #   光谱分支: [B,160,128,128] * [B,160,128,128] → [B,160,128,128]
        #   空间分支: [B,40,256,256]  * [B,40,256,256]  → [B,40,256,256]
        x = self.conv(x) * self.linear(x)
        # 光谱分支输出: [B, 160, 128, 128]
        # 空间分支输出: [B, 40, 256, 256]

        # ---- CNN格式 → 序列格式 (逆转换) ----
        # view(B, -1, H*W): 将2D空间(H,W)重新压平为1D(L)
        #   光谱分支: [B,160,128,128] → [B,160,16384]
        #   空间分支: [B,40,256,256]  → [B,40,65536]
        #
        # permute(0,2,1): 恢复Transformer标准格式 [B, 空间位置, 特征通道]
        #   光谱分支: [B,160,16384] → [B,16384,160]
        #   空间分支: [B,40,65536]  → [B,65536,40]
        #
        # contiguous(): 同上，permute后内存不连续需重排
        x = x.view(B, -1, H*W).permute(0,2,1).contiguous()
        # 光谱分支输出: [B, 16384, 160]  (= 输入形状)
        # 空间分支输出: [B, 65536, 40]   (= 输入形状)

        return x
        # 光谱分支返回: [B, 16384, 160]  — 富含空间+光谱信息的Q/V表示
        # 空间分支返回: [B, 65536, 40]


class Mlp(nn.Module):
    """ MLP-based Feed-Forward Network
    【论文对应】SSCL中的MLP部分，以及HDRTB中的FFN（Feed-Forward Network）。
    在SSCL中，输出经过SSFA（Spatial-Spectral Feature Aggregation）后通过MLP进行进一步变换。

    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """【HDRTB辅助函数】将特征图划分为非重叠的局部窗口，用于分层窗口自相关计算。
    在SSCL（论文 Section 2.3）中，输入特征首先被划分为窗口，然后在每个窗口内分别计算SpaSC和SpeSC。
    
    Args:
        x: (B, H, W, C)
        window_size (tuple): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """【HDRTB辅助函数】window_partition的逆操作，将窗口特征合并回完整特征图。
    在SSCL（论文 Section 2.3）中，SpaSC和SpeSC计算完成后需要通过此函数恢复原始分辨率。
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] * (window_size[0] * window_size[1]) / (H * W))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias (动态相对位置偏置)
    
    【论文对应】SpaSC（Spatial Self-Correlation）中的位置编码模块。
    在SpaSC计算空间相关性时，需要引入相对位置偏置来增强空间位置感知能力。
    本模块基于Crossformer实现，通过MLP将2D相对位置坐标映射为每个注意力头的偏置值。
    见论文 Figure 4 中 SpaSC 分支的位置偏置部分。

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of heads for spatial self-correlation.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

class SCC(nn.Module):
    """ Spatial-Channel Correlation (SCC) / Spatial-Spectral Correlation Layer (SSCL)
    
    【论文核心模块】对应论文 Section 2.3 "Spatial-Spectral Correlation Layer" 及 Figure 4。
    
    SSCL是HSSDCT的核心创新，将特征聚合解耦为空间和光谱两个相关性路径，实现线性复杂度：
      1. **SSFE (self.qv = DFE)**: 空间-光谱特征提取，生成Q和V表示
         - 卷积路径：局部空间编码
         - 线性路径：光谱编码/矩阵分解
      2. **SpaSC (spatial_self_correlation)**: 空间自相关
         - 公式：SpaSC(Q,V) = (QV^T / sqrt(d)) * V   （论文公式3）
         - 通过与空间压缩的Value token计算亲和力来建模长程空间依赖
      3. **SpeSC (channel_self_correlation)**: 光谱（通道）自相关
         - 公式：SpeSC(Q,V) = (Q^T V / HW) * V^T        （论文公式4）
         - 通过计算通道级亲和力保留精细的光谱特征签名
      4. **SSFA (self.proj)**: 空间-光谱特征聚合（Spatial-Spectral Feature Aggregation）
         - 将SpaSC和SpeSC的输出通过逐元素拼接后线性投影融合
    
    相比传统窗口自注意力的三大优势：
      (i) 复杂度随窗口大小线性增长（而非二次方），可使用更大的分层窗口
      (ii) 支持HDRTB中渐进式增大的窗口以获得更大感受野
      (iii) 显式建模光谱相关性，这在Transformer-based HSI融合方法中常被忽略

    Args:
        dim (int): Number of input channels.
        base_win_size (tuple[int]): The height and width of the base window.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of heads for spatial self-correlation.
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, base_win_size, window_size, num_heads, value_drop=0., proj_drop=0.):

        super().__init__()
        # parameters
        self.dim = dim
        self.window_size = window_size 
        self.num_heads = num_heads

        # feature projection
        self.qv = DFE(dim, dim)
        self.proj = nn.Linear(dim, dim)

        # dropout
        self.value_drop = nn.Dropout(value_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # base window size
        min_h = min(self.window_size[0], base_win_size[0])
        min_w = min(self.window_size[1], base_win_size[1])
        self.base_win_size = (min_h, min_w)

        # normalization factor and spatial linear layer for S-SC
        head_dim = dim // (2*num_heads)
        self.scale = head_dim
        self.spatial_linear = nn.Linear(self.window_size[0]*self.window_size[1] // (self.base_win_size[0]*self.base_win_size[1]), 1)

        # define a parameter table of relative position bias
        self.H_sp, self.W_sp = self.window_size
        self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
    
    def spatial_linear_projection(self, x):
        """【SpaSC辅助函数】空间线性投影：将Value特征在窗口内按base_win_size进行空间压缩。
        通过将每个base_window内的token聚合为一个表示，降低V的空间分辨率，
        从而使SpaSC的计算复杂度从O(N^2)降为O(N * N')，其中N' = H'W' << HW（论文公式3）。
        
        【调用来源】spatial_self_correlation() 在计算 corr_map = Q @ V^T / scale 前，
                   先对V进行空间压缩，减少相关性矩阵的列数。
        
        【真实输入形状】x 来自 window_partition 后的 V:
          光谱分支(dim=160,C=80): [256B,   8, 64, 80]   # win=(8,8)时: B'=256B, heads=8, L=64=8×8, C=80
          空间分支(dim=40,C=40):  [1024B,  5, 64, 40]   # win=(8,8)时: B'=1024B,heads=5, L=64=8×8, C=40
        
        【不同窗口大小下的行为】(base_win_size固定为(8,8)):
          win=(4,4):  不调用此函数(window<base,无需压缩)
          win=(8,8):  无实际压缩(map==win, 每个base_window=1个token, linear输入维=1→1)
          win=(16,16): 2×2压缩(每4个token聚合为1个, linear输入维=4→1)
          win=(32,32): 4×4压缩(每16个token聚合为1个, linear输入维=16→1)
        
        【返回值】压缩后的V: [B', num_heads, base_win_area=64, C]
        """
        # ===== 第1步: 解包输入张量的各个维度 =====
        B, num_h, L, C = x.shape
        # 光谱分支: B=256B, num_h=8,    L=64(=8×8), C=80
        # 空间分支: B=1024B,num_h=5,    L=64(=8×8), C=40
        # 注: L = window_size[0] * window_size[1]，即一个窗口内的token总数

        # ===== 第2步: 获取当前层的窗口尺寸和基础窗口尺寸 =====
        H, W = self.window_size       # 当前层窗口的高和宽 (总是正方形)
        map_H, map_W = self.base_win_size  # 基础窗口(8,8)，用于划分压缩单元
        # win=(8,8):   H,W=8,8     map_H,map_W=8,8      → 每个map_cell包含 1×1=1 个token
        # win=(16,16): H,W=16,16   map_H,map_W=8,8      → 每个map_cell包含 2×2=4 个token
        # win=(32,32): H,W=32,32   map_H,map_W=8,8      → 每个map_cell包含 4×4=16个token

        # ===== 第3步: 重塑张量结构，按base_win_size分组 =====
        # 将L=H×W的序列展成2D网格，再按base_win_size划分为子区域
        x = x.view(B, num_h, map_H, H//map_H, map_W, W//map_W, C)
        # view后形状(以win=(16,16)为例):
        #   [256B, 8, 8, 2, 8, 2, 80]
        #         ↑   ↑   ↑   ↑  ↑
        #         |   |   |   |  └─ C通道(不变)
        #         |   |   |   └──── W方向上每个map_cell内的分块数 = W//map_W = 16//8 = 2
        #         |   |   └──────── map_W方向的格子数 = 8
        #         |   └──────────── H方向上每个map_cell内的分块数 = H//map_H = 16//8 = 2
        #         └──────────────── map_H方向的格子数 = 8
        #
        # win=(8,8)时: [256B, 8, 8, 1, 8, 1, 80]  (无实际分组，每组只有1个token)
        # win=(32,32)时:[256B, 8, 8, 4, 8, 4, 80]  (4×4分组，每组16个token)

        # ===== 第4步: 调整维度顺序，把需要聚合的维度放到最后 =====
        x = x.permute(0, 1, 2, 4, 6, 3, 5)
        # permute后形状(win=(16,16)):
        #   [256B, 8, 8, 8, 80, 2, 2]
        #         ↑   ↑  ↑  ↑  ↑   ↑  ↑
        #         |   |  |  |  |   |  └─ W方向分块索引(要聚合)
        #         |   |  |  |  |   └──── H方向分块索引(要聚合)
        #         |   |  |  |  └──────── C通道
        #         |   |  |  └─────────── map_W格子数
        #         |   |  └────────────── map_H格子数
        #         |   └───────────────── num_heads
        #         └───────────────────── B'(batch of windows)

        # ===== 第5步: 保证内存连续并展平最后两个聚合维度 =====
        x = x.contiguous().view(B, num_h, map_H * map_W, C, -1)
        # contiguous(): permute后内存不连续，必须调用才能view()
        # 最后维度 -1 自动推断 = (H//map_H) * (W//map_W)，即每个map_cell内的token数量
        # view后形状:
        #   win=(8,8):   [256B, 8, 64, 80, 1]     # 64=8×8个map_cell, 每个1个token
        #   win=(16,16): [256B, 8, 64, 80, 4]     # 64=8×8个map_cell, 每个4个token(2×2)
        #   win=(32,32): [256B, 8, 64, 80, 16]    # 64=8×8个map_cell, 每个16个token(4×4)

        # ===== 第6步: 线性投影聚合 —— 核心压缩操作 =====
        x = self.spatial_linear(x)
        # spatial_linear = nn.Linear(in_features=(H*W)//(map_H*map_W), out_features=1)
        # 对最后一个维度做线性变换: 将每个map_cell内的K个token聚合为1个表示
        #   win=(8,8):   Linear(1→1): [256B, 8, 64, 80, 1]   → 实际上是恒等映射(或轻微变换)
        #   win=(16,16): Linear(4→1): [256B, 8, 64, 80, 4]   → [256B, 8, 64, 80, 1]  (4合1)
        #   win=(32,32): Linear(16→1):[256B, 8, 64, 80, 16]  → [256B, 8, 64, 80, 1]  (16合1)
        # 
        # 【作用】可学习的聚合函数: 将base_window区域内所有token的特征加权求和为一个向量
        #         类似于局部池化操作，但是可训练的参数化版本

        # ===== 第7步: 去掉多余的维度，返回压缩结果 =====
        x = x.view(B, num_h, map_H * map_W, C)
        # 去掉最后长度为1的维度
        # 最终返回形状(两种分支):
        #   光谱分支: [256B,   8, 64, 80]   # 64=8×8=base_win_area, C=dim/2=80
        #   空间分支: [1024B,  5, 64, 40]   # 64=8×8=base_win_area, C=dim/2=40
        #
        # ★ 关键: 输出的第3维恒为 base_win_area(64)，与原始window_size无关!
        #   这就是"空间压缩"的含义: 无论原始窗口多大，都压缩到base_win_size的粒度
        return x
    
    def spatial_self_correlation(self, q, v):
        """【SSCL - SpaSC】空间自相关 (Spatial Self-Correlation)
        
        【论文对应】论文 Section 2.3 公式(3): SpaSC(Q,V) = (QV^T / sqrt(d)) * V
        
        真实调用场景：
          光谱分支: q,v=[256B, 8, 64, 10], dim=160, win=(8,8), base_win=(8,8)
          空间分支: q,v=[1024B, 5, 64, 4], dim=40, win=(8,8), base_win=(8,8)
          (当win>base时: 如win=(16,16),base=(8,8)，V被压缩)
        
        在分层窗口内，通过将Query与空间压缩后的Value token进行相关性计算，
        实现高效的长程空间上下文建模。关键步骤：
          1. 对V进行空间线性投影（spatial_linear_projection），压缩空间维度
          2. 计算相关性图: corr_map = Q @ V_compressed^T / scale
          3. 添加可学习的动态相对位置偏置（DynamicPosBias）增强位置感知
          4. 通过相关性图对压缩后的V加权得到空间增强特征
        
        这种设计避免了标准自注意力的二次复杂度，使得可以使用更大的分层窗口。
        """
        
        # ---- Step 1: 解包输入形状 ----
        # 光谱分支: B=256B(256个窗×batch), num_head=8, L=64(=8×8), C=10(head_dim)
        # 空间分支: B=1024B(1024个窗×batch), num_head=5, L=64(=8×8), C=4(head_dim)
        B, num_head, L, C = q.shape

        # ---- Step 2: V的空间压缩投影（核心降复杂度操作）----
        # 将每个窗口内的Value从L个token压缩到H'W'个token
        # 内部流程:
        #   将[wh,ww]的窗口按[map_H,map_W]划分为子区域，每区域聚合为一个token
        #   通过Linear层加权求和实现聚合
        #
        # 【5种窗口大小对应的压缩行为】(base_win_size=(8,8)为全局基准):
        #
        # ① win=(4,4) < base:  self.base_win_size=min(4,8)=(4,4) → win==base, 无实际压缩
        #    L=16, Linear(1→1): 每个token独立，形状不变
        #    光谱分支: [1024B,8,16,80] → [1024B,8,16,80] (16=4×4)
        #    空间分支: [4096B,5,16,40] → [4096B,5,16,40] (16=4×4)
        #
        # ② win=(8,8) == base: 压缩比=1，无实际压缩
        #    L=64, Linear(1→1): 每个token独立，形状不变
        #    光谱分支: [256B,8,64,80] → [256B,8,64,80] (64=8×8)
        #    空间分支: [1024B,5,64,40] → [1024B,5,64,40] (64=8×8)
        #
        # ③ win=(16,16) > base: 压缩比=4x (每2×2的4个token聚合为1个)
        #    L=256→64, Linear(4→1)
        #    光谱分支: [256B,8,256,80] → [256B,8,64,80]
        #    空间分支: [1024B,5,256,40] → [1024B,5,64,40]
        #
        # ④ win=(32,32) > base: 压缩比=16x (每4×4的16个token聚合为1个)
        #    L=1024→64, Linear(16→1)
        #    光谱分支: [16B,8,1024,80] → [16B,8,64,80]
        #    空间分支: [64B,5,1024,40] → [64B,5,64,40]
        #
        # ★ 核心结论: 输出第3维恒为 self.base_win_size的面积!
        #   win≥(8,8)时: base=(8,8), 输出L=64
        #   win=(4,4)时: base=(4,4), 输出L=16
        v = self.spatial_linear_projection(v)
        # ★ 注意: 以下输出形状随window_size而变化! (以光谱分支为例)
        #   win=(4,4):  [1024B, 8,  16, 80]   # base=(4,4), L=4×4=16
        #   win=(8,8):  [256B,  8,  64, 80]   # base=(8,8), L=8×8=64
        #   win=(16,16):[256B,  8,  64, 80]   # base=(8,8), L=8×8=64 (压缩后)
        #   win=(32,32):[16B,    8,  64, 80]   # base=(8,8), L=8×8=64 (压缩后)
        #
        # 空间分支同理(C=40):
        #   win=(4,4):  [4096B, 5, 16, 40]
        #   win=(8,8):  [1024B, 5, 64, 40]
        #   win=(16,16):[1024B, 5, 64, 40]
        #   win=(32,32):[64B,   5, 64, 40]

        # ---- Step 3: 计算相关性图 corr_map = Q @ V^T / scale ----
        # transpose(-2,-1): 将V的最后两维转置 [B',head,C,L'] 用于矩阵乘法
        #   光谱分支: [256B,8,64,10] → [256B,8,10,64]
        #   空间分支: [1024B,5,64,4] → [1024B,5,4,64]
        #
        # q @ v^T: 批量矩阵乘法，计算Query与压缩后Value的相关性
        #   光谱分支: [256B,8,64,10] @ [256B,8,10,64] → [256B, 8, 64, 64]
        #             (64个token对64个token的相关性分数)
        #   空间分支: [1024B,5,64,4] @ [1024B,5,4,64] → [1024B, 5, 64, 64]
        #
        # / self.scale: 缩放因子防止数值过大 (scale=head_dim)
        #   光谱分支: /sqrt(10);  空间分支: /sqrt(4)=/2
        corr_map = (q @ v.transpose(-2,-1)) / self.scale
        # 光谱分支输出: [256B, 8, 64, 64]   (相关性图: 64×64亲和力矩阵)
        # 空间分支输出: [1024B, 5, 64, 64]

        # ---- Step 4: 添加动态相对位置偏置 (DynamicPosBias) ----
        # 目的: 让模型知道"token之间的距离"，相同内容在不同位置应有不同权重
        
        # 4a. 生成位置坐标的mother-set（所有可能的相对位置坐标）
        # arange: 产生 [-H_sp+1, ..., H_sp-1] 的坐标范围
        #   光谱/空间分支(win=8): H_sp=8 → [-7, -6, ..., 0, ..., 6, 7], 长度15
        position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=v.device)    # 长度: 2*H_sp-1=15
        position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=v.device)    # 长度: 2*W_sp-1=15
        
        # meshgrid + stack: 生成2D坐标网格，shape=[2, 15, 15]
        biases = torch.stack(torch.meshgrid(position_bias_h, position_bias_w, indexing='ij'))  # [2, 15, 15]
        
        # flatten+transpose: 转为坐标点列表 shape=[N_pos, 2]，每个元素是(dy,dx)
        #   225个可能的相对位置坐标 (15×15)
        rpe_biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # [225, 2]
        
        # pos网络: MLP(Linear→LN→ReLU→Linear→LN→ReLU→Linear→LN→ReLU→Linear→heads)
        # 将2D坐标映射为num_heads维的位置偏置向量
        #   光谱分支: [225, 2] → [225, 8]
        #   空间分支: [225, 2] → [225, 5]
        pos = self.pos(rpe_biases)  # [225, num_heads]

        # 4b. 为当前窗口内的每对token选择对应的位置偏置
        # 生成绝对坐标网格: coords_h=[0..7], coords_w=[0..7]
        coords_h = torch.arange(self.H_sp, device=v.device)  # [8]: [0,1,2,...,7]
        coords_w = torch.arange(self.W_sp, device=v.device)  # [8]: [0,1,2,...,7]
        
        # meshgrid: 生成2D坐标网格 shape=[2, 8, 8]
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2, 8, 8]
        
        # flatten: 展平为64个位置的坐标列表 shape=[2, 64]
        coords_flatten = torch.flatten(coords, 1)  # [2, 64]
        
        # 计算所有token对的相对坐标 (广播相减)
        # [:,:,None]→[2,64,1] 减 [:,None,:]→[2,1,64] → [2,64,64]
        # relative_coords[b,i,j] = 第j个token相对于第i个token的(dh,dw)偏移量
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, 64, 64]
        
        # permute: 调整为[token_i, token_j, (dh,dw)]格式
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [64, 64, 2]
        
        # 将相对坐标从[-7,+7]偏移到[0,14]的非负索引范围 (方便查表)
        #   dh: 加(H_sp-1)=7 → [-7..7] → [0..14]
        #   dw: 加(W_sp-1)=7 → [-7..7] → [0..14]
        relative_coords[:, :, 0] += self.H_sp - 1  # [64, 64, 0] += 7
        relative_coords[:, :, 1] += self.W_sp - 1  # [64, 64, 1] += 7
        
        # 将2D索引展平为1D: idx = dh * (2*W-1) + dw
        #   2*W_sp-1=15, 最终idx范围 [0, 224] (共225个值)
        relative_coords[:, :, 0] *= 2 * self.W_sp - 1  # [64, 64, 0] *= 15
        relative_position_index = relative_coords.sum(-1)  # [64, 64] 整数索引
        
        # 用索引从pos表中选择对应的位置偏置
        # view reshape: [64, 64] → [64, base_h, 64/base_h, base_w, 64/base_w, num_heads]
        #   光谱分支(base=(8,8)): [64, 8, 8, 1, 8, 1, 8]
        #   即 [Wh*Ww, bh, Wh/bh, bw, Ww/bw, nH] = [64, 8, 8, 1, 8, 1, 8]
        relative_position_bias = pos[relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1], self.base_win_size[0],
            self.window_size[0]//self.base_win_size[0], self.base_win_size[1],
            self.window_size[1]//self.base_win_size[1], -1)
        # 光谱分支: [64, 8, 8, 1, 8, 1, 8];  空间分支: [64, 8, 8, 1, 8, 1, 5]
        
        # permute+view: 重排维度并取均值，得到最终位置偏置 [num_heads, L, L']
        #   光谱分支: → [8, 64, 64]
        #   空间分支: → [5, 64, 64]
        relative_position_bias = relative_position_bias.permute(0,1,3,5,2,4).contiguous().view(
            self.window_size[0]*self.window_size[1], self.base_win_size[0]*self.base_win_size[1], self.num_heads, -1).mean(-1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # 光谱分支输出: [8, 64, 64];  空间分支输出: [5, 64, 64]
        
        # 将位置偏置加到相关性图上 (unsqueeze(0)添加batch维度)
        #   光谱分支: [256B,8,64,64] + [1,8,64,64] → [256B,8,64,64]
        #   空间分支: [1024B,5,64,64] + [1,5,64,64] → [1024B,5,64,64]
        corr_map = corr_map + relative_position_bias.unsqueeze(0)

        # ---- Step 5: 通过相关性图对V加权，得到空间自相关输出 ----
        # Dropout正则化 (训练时随机置零部分Value)
        # 形状不变: 光谱分支[256B,8,64,10]; 空间分支[1024B,5,64,4]
        v_drop = self.value_drop(v)
        
        # corr_map @ V': 相关性图作为权重对压缩后的V加权聚合
        #   光谱分支: [256B,8,64,64] @ [256B,8,64,10] → [256B, 8, 64, 10]
        #   空间分支: [1024B,5,64,64] @ [1024B,5,64,4] → [1024B, 5, 64, 4]
        x = (corr_map @ v_drop).permute(0,2,1,3).contiguous().view(B, L, -1)
        # permute(0,2,1,3): [B', head, L, C] → [B', L, head, C]
        # view(B, L, -1): 合并head和C维度
        #   光谱分支: [256B,8,64,10] → [256B,64,80]  (8头×10dim=80=C/2)
        #   空间分支: [1024B,5,64,4] → [1024B,64,20]  (5头×4dim=20=C/2)

        return x
        # 光谱分支返回: [256B, 64, 80]   (通道减半: 160/2=80)
        # 空间分支返回: [1024B, 64, 20] (通道减半: 40/2=20)
    
    def channel_self_correlation(self, q, v):
        """【SSCL - SpeSC】光谱/通道自相关 (Spectral Self-Correlation)
        
        【论文对应】论文 Section 2.3 公式(4): SpeSC(Q,V) = (Q^T V / HW) * V^T
        
        与SpaSC互补，SpeSC在光谱（通道）维度上建模相关性：
          - 采用单头策略，将所有头的Q和V在通道维度上拼接
          - 计算通道级亲和力矩阵: corr_map = Q^T V / L（L=HW为空间 token 数）
          - 通过通道级相关性对V转置后加权，保留精细光谱签名
        
        这对于高光谱图像融合至关重要，因为高光谱数据的核心价值在于其精细的光谱分辨能力。
        """
        
        B, num_head, L, C = q.shape

        # apply single head strategy - 采用单头策略：将所有注意力头的Q/V在通道维拼接
        q = q.permute(0,2,1,3).contiguous().view(B, L, num_head*C)
        v = v.permute(0,2,1,3).contiguous().view(B, L, num_head*C)

        # compute correlation map: Q^T V / HW，对应论文公式(4)
        corr_map = (q.transpose(-2,-1) @ v) / L
        
        # transformation: corr_map @ V^T，得到光谱自相关输出
        v_drop = self.value_drop(v)
        x = (corr_map @ v_drop.transpose(-2,-1)).permute(0,2,1).contiguous().view(B, L, -1)

        return x

    def forward(self, x):
        """【SSCL前向传播】完整执行空间-光谱相关层计算流程
        
        真实调用场景：
          光谱分支: x=[B, 128, 128, 160], dim=160, win=(8,8), num_heads=8
          空间分支: x=[B, 256, 256, 40],  dim=40,  win=(8,8), num_heads=5
        
        数据流（对应论文 Figure 4）:
          Input F -> [SSFE/DFE] -> Q,V -> [Split] -> Q,V (各半通道)
            -> [SpaSC分支] -> x_spatial (C/2维)
            -> [SpeSC分支] -> x_channel (C/2维)
            -> [Concat] -> [SSFA/Linear proj] -> Output
        """
        # ---- Step 0: 解包输入形状 ----
        # 光谱分支: [B, 128, 128, 160];  空间分支: [B, 256, 256, 40]
        xB, xH, xW, xC = x.shape

        # ---- Step 1: SSFE/DFE 双特征提取，生成Q和V表示 ----
        # ┌──────────────────────────────────────────────────────────────────┐
        # │ DFE内部流程（self.qv）:                                           │
        # │   输入: x.view(xB,-1,xC)=[xB, xH*xW, xC]                          │
        # │     光谱分支: [B, 16384, 160]                                      │
        # │     空间分支: [B, 65536, 40]                                       │
        # │                                                                  │
        # │   permute+view转Conv2d格式: [B, C, H, W]                           │
        # │     光谱分支: [B, 160, 128, 128]                                   │
        # │     空间分支: [B, 40, 256, 256]                                    │
        # │                                                                  │
        # │   卷积分支(conv):                                                  │
        # │     光谱分支: Conv2d(160→32)→LReLU→3×3(32→32)→LReLU→Conv2d(32→160)│
        # │       输出: [B, 160, 128, 128]                                     │
        # │     空间分支: Conv2d(40→8)→LReLU→3×3(8→8)→LReLU→Conv2d(8→40)      │
        # │       输出: [B, 40, 256, 256]                                      │
        # │                                                                  │
        # │   线性分支(linear):                                                │
        # │     光谱分支: Conv2d(160→160, k=1) → [B, 160, 128, 128]           │
        # │     空间分支: Conv2d(40→40, k=1)   → [B, 40, 256, 256]            │
        # │                                                                  │
        # │   融合: conv * linear（逐元素相乘，门控融合）                       │
        # │     光谱分支: [B, 160, 128, 128]                                   │
        # │     空间分支: [B, 40, 256, 256]                                    │
        # │                                                                  │
        # │   reshape回序列格式并view为图像格式:                                │
        # │     光谱分支: [B, 128, 128, 160]                                   │
        # │     空间分支: [B, 256, 256, 40]                                    │
        # └──────────────────────────────────────────────────────────────────┘
        qv = self.qv(x.view(xB, -1, xC), (xH, xW)).view(xB, xH, xW, xC)
        # 先转换为token格式，然后经过SSFE处理之后再转换为图片格式
        # 光谱分支输出: [B, 128, 128, 160]
        # 空间分支输出: [B, 256, 256, 40]

        # ---- Step 2: Window Partition 划分非重叠窗口 ----
        # 将特征图按window_size划分为局部窗口，每个窗口内独立计算相关性
        # 光谱分支(win=(8,8)): [B, 128, 128, 160] → [B*(16*16), 8, 8, 160] = [256B, 8, 8, 160]
        #   其中 128/8=16, 共16×16=256个窗口
        # 空间分支(win=(8,8)): [B, 256, 256, 40]  → [B*(32*32), 8, 8, 40]  = [1024B, 8, 8, 40]
        #   其中 256/8=32, 共32×32=1024个窗口
        qv = window_partition(qv, self.window_size)

        # 将每个窗口展平为序列: [num_win*B, wh*ww, C]
        # 光谱分支: [256B, 64, 160]  (64=8*8)
        # 空间分支: [1024B, 64, 40] (64=8*8)
        qv = qv.view(-1, self.window_size[0]*self.window_size[1], xC)

        # ---- Step 3: QV 分割 —— 将特征沿通道维均分为Q和V两部分 ----
        # 解包当前形状
        # 光谱分支: B'=256B, L=64, C=160
        # 空间分支: B'=1024B, L=64, C=40
        B, L, C = qv.shape

        # reshape为多头格式: [B', L, 2, num_heads, head_dim]
        # head_dim = C // (2 * num_heads)
        #   光谱分支: head_dim = 160 // 16 = 10; 形状: [256B, 64, 2, 8, 10]
        #   空间分支: head_dim = 40 // 10 = 4;   形状: [1024B, 64, 2, 5, 4]
        qv = qv.view(B, L, 2, self.num_heads, C // (2*self.num_heads))

        # permute将Q/V维度提到最前: [2, B', num_heads, L, head_dim]
        # 光谱分支: [2, 256B, 8, 64, 10]
        # 空间分支: [2, 1024B, 5, 64, 4]
        qv = qv.permute(2, 0, 3, 1, 4).contiguous()

        # 分离为Q和V
        # 光谱分支: q=v=[256B, 8, 64, 10]
        # 空间分支: q=v=[1024B, 5, 64, 4]
        q, v = qv[0], qv[1]  # [B, num_heads, L, head_dim]

        # ---- Step 4: SpaSC 空间自相关 (论文公式3) ----
        # 公式: SpaSC(Q,V) = (Q @ V_compressed^T / sqrt(d)) * V_compressed
        # 内部流程:
        #   ① spatial_linear_projection: 将V从[L,head_d]压缩到[H'W',head_d]
        #      光谱分支: V从[256B,8,64,10]压缩到[256B,8,64,10] (win==base_win时不变)
        #      若win>base: 如win=(16,16),base=(8,8)则V从[256B,8,256,10]压缩到[256B,8,64,10] (4×4→1)
        #   ② corr_map = Q @ V^T / scale
        #      光谱分支: [256B,8,64,10] @ [256B,8,10,64] / sqrt(10) → [256B, 8, 64, 64]
        #   ③ 添加DynamicPosBias动态相对位置偏置
        #   ④ corr_map @ V → 空间加权输出
        #
        # 输出: [B', L, C/2] （通道减半，因为空间相关只产生一半特征）
        # 光谱分支: [256B, 64, 80]   (80=160/2)
        # 空间分支: [1024B, 64, 20]  (20=40/2)
        x_spatial = self.spatial_self_correlation(q, v)

        # reshape回窗口格式以便后续merge
        # 光谱分支: [256B, 64, 80] → [256B, 8, 8, 80]
        # 空间分支: [1024B, 64, 20] → [1024B, 8, 8, 20]
        x_spatial = x_spatial.view(-1, self.window_size[0], self.window_size[1], C//2)

        # window_reverse: 合并窗口恢复完整特征图
        # 光谱分支: [256B, 8, 8, 80] → [B, 128, 128, 80]
        # 空间分支: [1024B, 8, 8, 20] → [B, 256, 256, 20]
        x_spatial = window_reverse(x_spatial, (self.window_size[0],self.window_size[1]), xH, xW)

        # ---- Step 5: SpeSC 光谱/通道自相关 (论文公式4) ----
        # 公式: SpeSC(Q,V) = (Q^T @ V / HW) * V^T
        # 内部流程:
        #   ① 单头策略: 将所有头的Q/V在通道维拼接
        #      光谱分支: [256B,8,64,10]→permute→[256B,64,80] (8头×10dim=80ch)
        #      空间分支: [1024B,5,64,4] →permute→[1024B,64,20](5头×4dim=20ch)
        #   ② corr_map = Q^T @ V / L (L=HW=64, 空间token数)
        #      光谱分支: [256B,80,64] @ [256B,64,20] / 64 → [256B, 80, 20]
        #   ③ corr_map @ V^T → 通道级加权输出
        #      [256B, 80, 20] @ [256B, 20, 64] → [256B, 80, 64]
        #
        # 输出: [B', L, C/2] （与SpaSC互补的另一半光谱特征）
        # 光谱分支: [256B, 64, 80]
        # 空间分支: [1024B, 64, 20]
        x_channel = self.channel_self_correlation(q, v)

        # reshape + window_reverse (同SpaSC)
        # 光谱分支: [256B, 64, 80] → [256B, 8, 8, 80] → [B, 128, 128, 80]
        # 空间分支: [1024B, 64, 20] → [1024B, 8, 8, 20] → [B, 256, 256, 20]
        x_channel = x_channel.view(-1, self.window_size[0], self.window_size[1], C//2)
        x_channel = window_reverse(x_channel, (self.window_size[0], self.window_size[1]), xH, xW)

        # ---- Step 6: SSFA 空间-光谱特征聚合 ----
        # 拼接两个互补的特征（通道维拼接，恢复原始通道数）
        # 光谱分支: cat([B,128,128,80], [B,128,128,80], dim=-1) → [B, 128, 128, 160]
        # 空间分支: cat([B,256,256,20], [B,256,256,20], dim=-1) → [B, 256, 256, 40]
        x = torch.cat([x_spatial, x_channel], -1)

        # Linear投影 + Dropout: 最终的特征变换和正则化
        # proj = nn.Linear(dim, dim)，将融合后的特征线性映射
        # 光谱分支: Linear(160→160): [B, 128, 128, 160] → [B, 128, 128, 160]
        # 空间分支: Linear(40→40):   [B, 256, 256, 40]  → [B, 256, 256, 40]
        x = self.proj_drop(self.proj(x))

        # 返回SSCL最终输出（与输入形状一致）
        # 光谱分支: [B, 128, 128, 160]
        # 空间分支: [B, 256, 256, 40]
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class HierarchicalTransformerBlock(nn.Module):
    """ Hierarchical Transformer Block (分层Transformer块)
    
    【论文对应】HDRTB (Hierarchical Dense-Residue Transformer Block) 的核心构建单元，
    对应论文 Section 2.2 及 Figure 2/3。
    
    本模块将SCC（即SSCL）封装为标准Transformer Block的形式，包含：
      - LayerNorm + SSCL（空间-光谱相关）+ 残差连接
      - LayerNorm + MLP(FFN) + 残差连接
    支持可变的分层窗口大小（hierarchical window size），这是HDRTB实现渐进式感受野扩大的基础。
    
    HDRTB的两个关键设计理念（论文 Section 2.2）：
      1. **分层窗口（Hierarchical Windows）**：窗口大小随深度递增（如 {4,8,16,16}），
         浅层捕获局部纹理，深层聚合全局上下文
      2. **密集残差连接（Dense-Residue Connections）**：各层特征通过拼接+1x1卷积融合，
         有效扩大感受野且不显著增加复杂度（公式2: F_out = F_in + gamma * Conv_1x1(Cat(F1,F2,F3))）
    
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of heads for spatial self-correlation.
        base_win_size (tuple[int]): The height and width of the base window.
        window_size (tuple[int]): The height and width of the window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, base_win_size, window_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size 
        self.mlp_ratio = mlp_ratio

        # check window size
        if (window_size[0] > base_win_size[0]) and (window_size[1] > base_win_size[1]):
            assert window_size[0] % base_win_size[0] == 0, "please ensure the window size is smaller than or divisible by the base window size"
            assert window_size[1] % base_win_size[1] == 0, "please ensure the window size is smaller than or divisible by the base window size"


        self.norm1 = norm_layer(dim)
        self.correlation = SCC(
            dim, base_win_size=base_win_size, window_size=self.window_size, num_heads=num_heads,
            value_drop=value_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def check_image_size(self, x, win_size):
        #   1. permute(0,3,1,2): [B,H,W,C] → [B,C,H,W]（Conv2d格式）
        #   2. 计算mod_pad_h = (win_h - H%win_h) % win_h（需要填充的高度）
        #   3. 计算mod_pad_w = (win_w - W%win_w) % win_w（需要填充的宽度）
        #   4. F.pad(..., 'reflect'): reflect模式边缘反射填充
        #   5. permute(0,2,3,1): [B,C,H+ph,W+pw] → [B,H+ph,W+pw,C]
        # 示例: H=128,W=128,win=(8,16) → 128%8=0,128%16=0 → 无需填充
        #       H=126,W=130,win=(8,16) → pad_h=(8-126%8)%8=6, pad_w=(16-130%16)%16=6
        x = x.permute(0,3,1,2).contiguous()
        _, _, h, w = x.size()
        mod_pad_h = (win_size[0] - h % win_size[0]) % win_size[0]
        mod_pad_w = (win_size[1] - w % win_size[1]) % win_size[1]
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = x.permute(0,2,3,1).contiguous()
        return x

    def forward(self, x, x_size, win_size):
        """【HDRTB子块前向传播】执行单层分层Transformer计算
        
        真实调用场景：
          光谱分支: x=[B, 16384, 160], x_size=(128,128), win_size=(4,4)/(8,8)/(16,16)/(32,32)
          空间分支: x=[B, 65536, 40],  x_size=(256,256), win_size=(4,4)/(8,8)/(16,16)/(32,32)
        
        Args:
            x: 输入序列特征
            x_size: 原始图像空间尺寸
            win_size: 当前层窗口大小（恒为正方形）
        
        Returns:
            x: 输出序列特征，形状与输入相同
        
        流程: 
          1. 将序列特征reshape为2D特征图
          2. padding确保尺寸能被window_size整除
          3. 通过SSCL（SCC）进行空间-光谱相关计算
          4. 去除padding，恢复原始分辨率
          5. LayerNorm + 残差连接（Post-Norm风格）
          6. FFN(MLP) + 残差连接
        
        对应论文 Figure 2 中 SSCL 内部的 iLayerNorm -> SSCL -> MLP 结构。
        """
        # 解包空间尺寸
        # 光谱分支: x_size=(128,128) → H=128, W=128
        # 空间分支: x_size=(256,256) → H=256, W=256
        H, W = x_size

        # 解包输入张量形状
        # 光谱分支: [B, 16384, 160] → B=B, L=16384(=128*128), C=160(=nf*2)
        # 空间分支: [B, 65536, 40]  → B=B, L=65536(=256*256), C=40 (=nf//2)
        B, L, C = x.shape

        # 保存输入用于残差连接（浅拷贝引用）
        # 光谱分支: [B, 16384, 160]
        # 空间分支: [B, 65536, 40]
        shortcut = x

        # 将序列格式 reshape 为2D特征图格式（用于窗口操作）
        # 光谱分支: [B, 16384, 160] → [B, 128, 128, 160]
        # 空间分支: [B, 65536, 40]  → [B, 256, 256, 40]
        x = x.view(B, H, W, C)

        # ---- Padding：确保空间尺寸能被win_size整除（窗口操作要求）----
        # check_image_size内部流程：
        #   permute→计算pad量→reflect padding→permute还原
        # 示例: H=128,W=128,win=(8,8) → 128%8=0无需padding → 形状不变
        #       H=130,W=126,win=(8,8) → pad_h=6,pad_w=2 → [B,136,128,C]
        # 光谱分支: [B, 128, 128, 160] → 通常无padding → [B, 128, 128, 160]
        # 空间分支: [B, 256, 256, 40]  → 通常无padding → [B, 256, 256, 40]
        x = self.check_image_size(x, win_size)

        # 记录padding后的实际尺寸（供unpad使用）
        # 通常H_pad==H, W_pad==W（因为128和256都是8的倍数）
        _, H_pad, W_pad, _ = x.shape

        # SCC/SSCL：核心的空间-光谱相关性计算（论文Section 2.3, 公式3&4）
        # 内部执行：SSFE(Q,V生成) → SpaSC(空间自相关) + SpeSC(通道自相关) → SSFA融合
        # 输入/输出形状不变（SCC内部保持[H,W,C]格式）
        # 光谱分支: [B, 128, 128, 160] → [B, 128, 128, 160]
        # 空间分支: [B, 256, 256, 40]  → [B, 256, 256, 40]
        x = self.correlation(x)

        # ---- Unpad：去除之前添加的padding，恢复原始空间分辨率 ----
        # 切片截取前H行、前W列，contiguous()保证内存连续
        # 光谱分支: [B, 128, 128, 160] → [B, 128, 128, 160]
        # 空间分支: [B, 256, 256, 40]  → [B, 256, 256, 40]
        x = x[:, :H, :W, :].contiguous()

        # ---- LayerNorm（Post-Norm风格：先变换再归一化）----
        # Step1: reshape回序列格式
        # 光谱分支: [B, 128, 128, 160] → [B, 16384, 160]
        # 空间分支: [B, 256, 256, 40]  → [B, 65536, 40]
        x = x.view(B, H * W, C)

        # Step2: LayerNorm在最后一个维度(C维)上归一化
        # norm1=LayerNorm(C)，对每个patch的特征向量做归一化（均值0方差1后仿射变换）
        # 光谱分支: [B, 16384, 160] → [B, 16384, 160]
        # 空间分支: [B, 65536, 40]  → [B, 65536, 40]
        x = self.norm1(x)

        # ---- 第一个残差连接：shortcut + drop_path(SSCL输出) ----
        # drop_path: 随机深度正则化（训练时以drop_path概率置零，推理时恒等映射）
        # 光谱分支: [B,16384,160] + [B,16384,160] → [B, 16384, 160]
        # 空间分支: [B,65536,40]  + [B,65536,40]  → [B, 65536, 40]
        x = shortcut + self.drop_path(x)  # 残差连接①

        # ---- FFN (Feed-Forward Network / MLP) 子层 ----
        # 完整流程: mlp(fc1→GELU→drop→fc2→drop) → norm2 → drop_path → 残差相加
        # ┌────────────────────────────────────────────────────────────┐
        # │ mlp内部结构（Mlp类）：                                       │
        # │   光谱分支: fc1=Linear(160→640), fc2=Linear(640→160)      │
        # │   空间分支: fc1=Linear(40→160),  fc2=Linear(160→40)       │
        # │   hidden_dim = dim * mlp_ratio = dim * 4                   │
        # │                                                              │
        # │ 光谱分支数据流:                                               │
        # │   [B,16384,160] → fc1 → [B,16384,640] → GELU              │
        # │   → [B,16384,640] → fc2 → [B,16384,160] → drop           │
        # │                                                              │
        # │ 空间分支数据流:                                               │
        # │   [B,65536,40] → fc1 → [B,65536,160] → GELU               │
        # │   → [B,65536,160] → fc2 → [B,65536,40] → drop            │
        # └────────────────────────────────────────────────────────────┘
        #
        # 完整链路: mlp → norm2 → drop_path → 残差相加
        # 光谱分支: [B, 16384, 160] + [B, 16384, 160] → [B, 16384, 160]
        # 空间分支: [B, 65536, 40]  + [B, 65536, 40]  → [B, 65536, 40]
        x = x + self.drop_path(self.norm2(self.mlp(x)))  # 残差连接②

        return x  # 形状与输入完全一致
        # 光谱分支返回: [B, 16384, 160]
        # 空间分支返回: [B, 65536, 40]

        return x  # → [B, L, C]  （形状与输入完全一致）

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


class MultiScaleFeatFusionBlock_Depthwise(nn.Module):
    """Multi-Scale Feature Fusion Block using Depthwise Convolutions."""
    def __init__(self, nf=64, gc=32, bias=False, groups=4):
        super(MultiScaleFeatFusionBlock_Depthwise, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias,dilation=1, groups=nf)
        self.conv2 = nn.Conv2d(nf + gc, nf + gc, 3, 1, 1, bias=bias,dilation=1, groups=nf + gc)
        self.conv3 = nn.Conv2d(nf + 2 * gc, nf + 2 * gc, 3, 1, 1, bias=bias,dilation=1, groups=nf + 2 * gc)
        self.conv4 = nn.Conv2d(nf + 3 * gc, nf + 3 * gc, 3, 1, 1, bias=bias,dilation=1, groups=nf + 3 * gc)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf + 4 * gc, 3, 1, 1, bias=bias,dilation=1, groups=nf + 4 * gc)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pointwise1 = nn.Conv2d(nf,gc,1,1,0,1,1,bias=bias)
        self.pointwise2 = nn.Conv2d(nf + gc,gc,1,1,0,1,1,bias=bias)
        self.pointwise3 = nn.Conv2d(nf + 2 * gc,gc,1,1,0,1,1,bias=bias)
        self.pointwise4 = nn.Conv2d(nf + 3 * gc,gc,1,1,0,1,1,bias=bias)
        self.pointwise5 = nn.Conv2d(nf + 4 * gc,nf,1,1,0,1,1,bias=bias)



        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        initialize_weights([self.pointwise1 ,self.pointwise2,self.pointwise3,self.pointwise4,self.pointwise5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.pointwise1(self.conv1(x)))
        x2 = self.lrelu(self.pointwise2(self.conv2(torch.cat((x, x1), 1))))
        x3 = self.lrelu(self.pointwise3(self.conv3(torch.cat((x, x1, x2), 1))))
        x4 = self.lrelu(self.pointwise4(self.conv4(torch.cat((x, x1, x2, x3), 1))))
        x5 = self.pointwise5(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))
        return x5 * 0.2 + x


class MultiScaleFeatFusionBlock(nn.Module):
    """Multi-Scale Feature Fusion Block."""

    def __init__(self, nf=64, gc=32, bias=True, groups=4):
        super(MultiScaleFeatFusionBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias, groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class MultiScaleFeatAggregation(nn.Module):
    """Multi-Scale Feature Aggregation Module."""

    def __init__(self, nf, gc=32, groups=4, mode=0):
        super(MultiScaleFeatAggregation, self).__init__()
        if mode ==0:
            self.MFB1 = MultiScaleFeatFusionBlock(nf, gc, groups=groups)
            self.MFB2 = MultiScaleFeatFusionBlock(nf, gc, groups=groups)
            self.MFB3 = MultiScaleFeatFusionBlock(nf, gc, groups=groups)
        elif mode ==1:
            self.MFB1 = MultiScaleFeatFusionBlock_Depthwise(nf, gc, groups=groups)
            self.MFB2 = MultiScaleFeatFusionBlock_Depthwise(nf, gc, groups=groups)
            self.MFB3 = MultiScaleFeatFusionBlock_Depthwise(nf, gc, groups=groups)

    def forward(self, x):
        out = self.MFB1(x)
        out = self.MFB2(out)
        out = self.MFB3(out)
        return out * 0.2 + x

class SwinBasedFeatFusionBlock(nn.Module):
    """【HDRTB完整实现】Swin-Based Feature Fusion Block (基于Swin的特征融合块)
    
    【论文对应】HDRTB (Hierarchical Dense-Residue Transformer Block)，论文 Section 2.2 及 Figure 2/3。
    
    这是HSSDCT中用于多尺度特征聚合的核心模块，实现了论文中描述的两大创新：
    
    1. **分层窗口（Hierarchical Windows）**：
       - 通过 hier_win_ratios 参数控制每层的窗口大小相对于 base_win_size 的比例
       - 默认 [0.5, 1, 2, 2] 对应4个渐进增大的窗口，例如 base=(8,8) 时窗口为 {(4,4), (8,8), (16,16), (16,16)}
       - 论文实验中使用的窗口大小为 {4, 8, 16, 16}
       - 浅层小窗口捕获细粒度局部纹理，深层大窗口建模全局语义结构
    
    2. **密集残差连接（Dense-Residue Connections）**：
       - 每层提取的特征 x_i 都与之前所有层特征拼接后送入下一层
       - 最终通过 1x1 卷积投影并乘以缩放因子 gamma=0.2 与输入残差相加
       - 公式：F_out = F_in + 0.2 * Conv_1x1(Cat(F_1, F_2, F_3, F_4))   （论文公式2）
       - 这避免了梯度消失问题并增强特征复用
    
    结构: swin1 -> adjust1 -> [cat(x,x1)] -> swin2 -> adjust2 -> [cat(x,x1,x2)] 
          -> swin3 -> adjust3 -> [cat(x,x1,x2,x3)] -> swin4(MLP ratio=1) -> adjust4
    其中每个swin是HierarchicalTransformerBlock（内含SSCL），每个adjust是1x1卷积通道调整层。
    
    注意: 第4个swin块的mlp_ratio=1（而非默认的4），用于轻量化最后的特征变换。
    """
    def __init__(self, dim, input_resolution, depth, num_heads, base_win_size, mlp_ratio, drop, value_drop, drop_path, norm_layer, gc, patch_size, img_size, hier_win_ratios=[0.5,1,2,2,4]):
        super(SwinBasedFeatFusionBlock, self).__init__()



        self.win_hs = [int(base_win_size[0] * ratio) for ratio in hier_win_ratios]
        self.win_ws = [int(base_win_size[1] * ratio) for ratio in hier_win_ratios]

        self.swin1 = HierarchicalTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                                          base_win_size=base_win_size, window_size=(self.win_hs[0], self.win_ws[0]),
                                          mlp_ratio=mlp_ratio,
                                          drop=drop, value_drop=value_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust1 = nn.Conv2d(dim, gc, 1) 
        
        self.swin2 = HierarchicalTransformerBlock(dim + gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + gc)%num_heads), base_win_size=base_win_size, window_size=(self.win_hs[1], self.win_ws[1]),
                                          mlp_ratio=mlp_ratio,
                                          drop=drop, value_drop=value_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust2 = nn.Conv2d(dim+gc, gc, 1) 
        
        self.swin3 = HierarchicalTransformerBlock(dim + 2 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 2*gc)%num_heads), base_win_size=base_win_size, window_size=(self.win_hs[2], self.win_ws[2]),
                                          mlp_ratio=mlp_ratio,
                                          drop=drop, value_drop=value_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust3 = nn.Conv2d(dim+gc*2, gc, 1) 
        
        self.swin4 = HierarchicalTransformerBlock(dim + 3 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 3*gc)%num_heads), base_win_size=base_win_size, window_size=(self.win_hs[3], self.win_ws[3]),
                                          mlp_ratio=1,
                                          drop=drop, value_drop=value_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust4 = nn.Conv2d(dim+gc*3, dim, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.pe = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.pue = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)
        
       

    def forward(self, x, xsize):
        """【HDRTB前向传播】执行密集残差特征融合
        
        数据流（对应论文 Figure 2 的 HDRTB 结构和 Figure 3(d)）:
          Input x 
            -> swin1(最小窗口) -> adjust1(降维到gc) -> LeakyReLU -> x1
            -> [cat(x, x1)] -> swin2(中等窗口) -> adjust2(降维到gc) -> LeakyReLU -> x2  
            -> [cat(x, x1, x2)] -> swin3(较大窗口) -> adjust3(降维到gc) -> LeakyReLU -> x3
            -> [cat(x, x1, x2, x3)] -> swin4(最大窗口, mlp_ratio=1) -> adjust4(恢复到dim) -> LeakyReLU -> x4
          Output = x4 * 0.2 + x   （残差连接，gamma=0.2，论文公式2）
        
        每层通过PatchEmbed/PatchUnEmbed在Transformer序列格式和图像格式间转换，
        以支持HierarchicalTransformerBlock的窗口操作。
        """
        x1 = self.pe(self.lrelu(self.adjust1(self.pue(self.swin1(x,xsize, (self.win_hs[0], self.win_ws[0])), xsize))))
        x2 = self.pe(self.lrelu(self.adjust2(self.pue(self.swin2(torch.cat((x, x1), -1), xsize, (self.win_hs[1], self.win_ws[1])), xsize))))
        x3 = self.pe(self.lrelu(self.adjust3(self.pue(self.swin3(torch.cat((x, x1, x2), -1), xsize, (self.win_hs[2], self.win_ws[2])), xsize))))
        x4 = self.pe(self.lrelu(self.adjust4(self.pue(self.swin4(torch.cat((x, x1, x2, x3), -1), xsize, (self.win_hs[3], self.win_ws[3])), xsize))))

        return x4 * 0.2 + x   

class SwinBasedFeatFusionBlock_final_block(nn.Module):
    """【最终融合HDRTB】基于Swin Transformer的最终特征融合块
    
    【论文对应】F_final 融合层中的 HDRTB，用于对融合后的光谱特征（YFD）进行精细化重建。
    
    与 SwinBasedFeatFusionBlock 的区别：
      - 使用标准 SwinTransformerBlock（带shifted window机制）而非 HierarchicalTransformerBlock
      - 支持 shift_size 参数实现窗口移位（SW-MSA），增强跨窗口信息交互
      - 同样采用密集残差连接结构（4个子块级联 + 残差输出）
    
    此模块位于双分支特征相加后的最终重建路径中（论文 Figure 1 中的 F_final 部分），
    对输出 HR-HSI 进行最后的细节恢复和重建。
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, gc, patch_size, img_size):
        super(SwinBasedFeatFusionBlock_final_block, self).__init__()

        self.swin1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                          num_heads=num_heads, window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust1 = nn.Conv2d(dim, gc, 1) 
        
        self.swin2 = SwinTransformerBlock(dim + gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + gc)%num_heads), window_size=window_size,
                                          shift_size=window_size//2,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust2 = nn.Conv2d(dim+gc, gc, 1) 
        
        self.swin3 = SwinTransformerBlock(dim + 2 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 2 * gc)%num_heads), window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust3 = nn.Conv2d(dim+gc*2, gc, 1) 
        
        self.swin4 = SwinTransformerBlock(dim + 3 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 3 * gc)%num_heads), window_size=window_size,
                                          shift_size=window_size//2,  # For first block
                                          mlp_ratio=1,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust4 = nn.Conv2d(dim+gc*3, dim, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.pe = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.pue = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, xsize):
        """【最终融合HDRTB前向传播】使用Swin Transformer（带窗口移位）进行密集残差特征融合
        与SwinBasedFeatFusionBlock结构类似，但使用shifted-window SwinTransformerBlock增强跨窗口交互。
        输出: x4 * 0.2 + x （残差连接）
        """
        x1 = self.pe(self.lrelu(self.adjust1(self.pue(self.swin1(x,xsize), xsize))))
        x2 = self.pe(self.lrelu(self.adjust2(self.pue(self.swin2(torch.cat((x, x1), -1), xsize), xsize))))
        x3 = self.pe(self.lrelu(self.adjust3(self.pue(self.swin3(torch.cat((x, x1, x2), -1), xsize), xsize))))
        x4 = self.pe           (self.adjust4(self.pue(self.swin4(torch.cat((x, x1, x2, x3), -1), xsize), xsize)))

        return x4 * 0.2 + x


class Mlp_swin(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_swin(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse_swin(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:

        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    
    【论文对应】用于最终重建层（SwinBasedFeatFusionBlock_final_block）中的标准窗口多头自注意力。
    注意: HDRTB的主体使用的是SSCL（SCC模块）而非此标准W-MSA，因为SSCL实现了线性复杂度的
    空间-光谱解耦相关。此W-MSA仅在最终的Swin-based融合块中使用，提供shifted-window机制。
    
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block (Swin Transformer块)
    
    【论文对应】用于最终重建层（SwinBasedFeatFusionBlock_final_block）中的标准Swin Transformer块。
    与HierarchicalTransformerBlock的区别：本模块使用标准的W-MSA/SW-MSA（二次复杂度），
    而非SSCL的线性复杂度空间-光谱解耦注意力。支持shifted window机制增强跨窗口交互。
    
    额外包含 HAI (Hyperprior-based Adaptive Initialization) 的 gamma 参数，
    通过可学习的缩放因子实现自适应残差连接。

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_swin(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        
        # HAI
        self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition_swin(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition_swin(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse_swin(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # HAI
        x = x + (shortcut * self.gamma)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, hier_win_ratios=[0.5,1,2,4,6,8]):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x,x_size)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # Flatten to [B, num_patches, C]
        if self.norm is not None:
            x = self.norm(x)  # Apply normalization
        return x


    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
class PatchUnEmbed(nn.Module):


    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size) 
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  

        self.in_chans = in_chans  
        self.embed_dim = embed_dim  

    def forward(self, x, x_size):
        B, HW, C = x.shape 
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])  
        return x


class YDCFN(nn.Module):
    """【论文核心网络】YDCFN - HSSDCT的整体双分支融合网络架构
    
    【论文对应】论文 Section 2.1 "Overall Framework" 及 Figure 1。
    
    HSSDCT的整体架构遵循双分支设计：
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    HSSDCT 整体架构 (Figure 1)                 │
    │                                                             │
    │   LR-HSI (低分辨率高光谱图像)          HR-MSI (高分辨率多光谱图像)  │
    │   64×64×172                           256×256×Mm (4或6波段)   │
    │        │                                     │                │
    │   [光谱分支 Spectral Branch]         [空间分支 Spatial Branch]  │
    │        │                                     │                │
    │   concat(LR-HSI, LR-MSI)               concat(HR-MSI, HR-HSI)  │
    │   Conv3x3 + LeakyReLU                 Conv3x3 + LeakyReLU      │
    │   Upsample ×2                         (无需上采样)              │
    │        │                                     │                │
    │   HDRTB ×2 (分层窗口特征提取)           HDRTB ×2                   │
    │   (SwinBasedFeatFusionBlock)          (SwinBasedFeatFusionBlock)│
    │        │                                     │                │
    │   Upsample ×2                          Conv3x3                  │
    │   Conv3x3                              LeakyReLU                │
    │   LeakyReLU                            Conv3x3                  │
    │        │                                     │                │
    │        └─────── element-wise add ─────────────┘                 │
    │                        │                                       │
    │              Conv_fuse (特征融合)                               │
    │                        │                                       │
    │              Final HDRTB (最终重建 F_final)                      │
    │              (SwinBasedFeatFusionBlock_final_block)             │
    │                        │                                       │
    │              Conv3x3 → HR-HSI 输出 (256×256×172)               │
    │                                                             │
    │  公式: Y* = F_final(F_spe + F_spa)     （论文公式1）           │
    └─────────────────────────────────────────────────────────────┘
    
    关键组件说明：
      - **光谱分支**：处理LR-HSI（富含光谱信息但空间分辨率低），通过上采样+HDRTB提取光谱特征 F_spe
      - **空间分支**：处理HR-MSI（空间分辨率高但波段有限），通过HDRTB提取空间特征 F_spa  
      - **特征融合**：两分支特征相加后通过卷积融合层得到初步重建结果YFD
      - **最终重建**：通过额外的HDRTB（使用Swin-based变体）进行精细化重建
      - LRMSI/HRHSI：分别从HRMSI下采样和LRHSI上采样获得，用于双分支的跨模态信息辅助
    """
    
    def make_layer(block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def __init__(self, in_nc=172,out_nc=172, nf=80, in_msi=4, gc=32, groups=4,
                img_size=128, patch_size=4, in_chans=172, embed_dim=32, 
                depths=[1,1], num_heads=[8,8],
                 window_size=[8,8], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, debug=False, hier_win_ratios=[0.5,1,2,4], **kwargs):
        """初始化HSSDCT双分支融合网络
        
        参数说明（对应论文实验设置）：
          - in_nc/out_nc: 输入/输出光谱波段数，默认172（AVIRIS数据集）
          - nf: 基础特征通道数（论文中对应特征图通道维度）
          - in_msi: MSI波段数，默认4（论文实验使用4-band或6-band HR-MSI）
          - gc: HDRTB中的growth channel数（密集连接中间通道），默认32
          - img_size: 空间分支输入图像尺寸，默认128
          - window_size: HDRTB的基础窗口大小，默认(8,8)，实际窗口通过hier_win_ratios缩放
          - hier_win_ratios: 分层窗口比例，默认[0.5,1,2,4]，
            对应窗口为base×ratio，如base=(8,8)时得到{(4,4),(8,8),(16,16),(32,32)}
            论文实验中HDRTB窗口设置为 {4, 8, 16, 16}
          - mlp_ratio: FFN隐藏层放大倍数，默认4
        """
        super(YDCFN, self).__init__()
        
        # ==================== 光谱分支 (Spectral Branch) ====================
        # 处理LR-HSI：富含172个光谱波段，空间分辨率低(64×64)
        self.debug = debug
        in_nc_group = groups
        if in_nc % groups != 0:
            in_nc_group = 1

        self.hsiconv1 = nn.Conv2d(in_nc+in_msi, nf*2, 3, 1, 1, bias=True, groups=in_nc_group)
        # 光谱分支首层卷积：输入为LR-HSI(172ch) + LR-MSI下采样版本(Mm ch)的拼接，输出nf*2通道特征
        self.hsiconvlast = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True, groups=groups)
        # 光谱分支末层卷积：将通道数从nf*2降到nf，与空间分支对齐
        self.up = torch.nn.Upsample(scale_factor=2)
        # 上采样层（×2）：LR-HSI需要两次上采样达到目标分辨率（64→128→256）
        
       
        self.hsifeat = nn.ModuleList()
        for _ in range(2):
            self.hsifeat.append(SwinBasedFeatFusionBlock(dim=nf*2, input_resolution=(8,8), depth=0,
                                 num_heads=8 - ((nf*2)%8), base_win_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 drop=drop_rate, value_drop=attn_drop_rate,
                                 drop_path=0, norm_layer=norm_layer,gc=gc, img_size=img_size//2, patch_size=patch_size, hier_win_ratios=hier_win_ratios))
        # 光谱分支HDRTB堆叠（×2个）：
        #   每个SwinBasedFeatFusionBlock包含4个HierarchicalTransformerBlock（内含SSCL）
        #   使用渐进增大的分层窗口进行多尺度光谱-空间特征提取
        #   dim=nf*2 (较大维度以保留丰富的光谱信息)

        # ==================== 空间分支 (Spatial Branch) ====================
        # 处理HR-MSI：空间分辨率高(256×256)，但波段有限(4或6个)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.msiconv1 = nn.Conv2d(in_msi+in_nc, nf//2, 3, 1, 1, bias=True)
        # 空间分支首层卷积：输入为HR-MSI(Mm ch) + LR-HSI上采样版本(172 ch)的拼接，输出nf//2通道
        self.msiconvlast = nn.Conv2d(nf//2, nf, 3, 1, 1, bias=True)
        # 空间分支末层卷积：将通道数从nf/2提升到nf，与光谱分支对齐
      
        self.msifeat = nn.ModuleList()
        for _ in range(2):
            self.msifeat.append(SwinBasedFeatFusionBlock(dim=nf//2, input_resolution=(4,4), depth=0,
                                 num_heads=8 - (nf//2%8), base_win_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 drop=drop_rate, value_drop=attn_drop_rate,
                                 drop_path=0, norm_layer=norm_layer,gc=gc, img_size=img_size, patch_size=patch_size, hier_win_ratios=hier_win_ratios))
        # 空间分支HDRTB堆叠（×2个）：
        #   与光谱分支类似但使用更小的特征维度(nf//2)，因为MSI波段较少
        #   input_resolution=(4,4)表示在256×256图像上使用更大的等效窗口

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        # ==================== 特征融合层 (Feature Fusion) ====================
        
        self.conv_fuse = nn.Sequential(nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True, groups=in_nc_group), 
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # 融合卷积层：将双分支拼接的nf通道特征映射到out_nc(172)通道
        # 对应论文公式1中的 F_final 之前的融合操作

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer= None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Reshape patches back to image format
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.final_blk = nn.ModuleList()
        for _ in range(1):
            self.final_blk.append( SwinBasedFeatFusionBlock_final_block(dim=out_nc, input_resolution=(4,4), depth=0,
                                 num_heads=8- (out_nc%8), window_size= 8, shift_size= 8//2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=0, norm_layer=norm_layer,gc=32, img_size=img_size, patch_size=patch_size))
        # 最终重建HDRTB（F_final）：
        #   使用SwinBasedFeatFusionBlock_final_block（基于Swin Transformer，支持shifted window）
        #   对融合后的172通道特征进行精细化重建
        #   输入维度为out_nc(172)，即直接在全光谱维度上操作

        self.norm = norm_layer(self.num_features)

        self.last = nn.Conv2d(out_nc, out_nc, 3, 1, 1, bias=False)
        # 最终重建卷积：3×3无偏置卷积层，生成最终HR-HSI输出
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward(self, lrhsi, hrmsi):
        """【YDCFN前向传播】执行完整的HSI融合流程（论文 Figure 1）
        
        Args:
            lrhsi: 低分辨率高光谱图像 (B, 172, H/4, W/4)，如 (B, 172, 64, 64)
            hrmsi: 高分辨率多光谱图像 (B, Mm, H, W)，如 (B, 4, 256, 256)，Mm=4或6
        
        Returns:
            co: 重建的高分辨率高光谱图像 (B, 172, H, W)，如 (B, 172, 256, 256)
        
        完整数据流:
          1. 准备跨模态辅助信息（LR-MSI和HR-HSI）
          2. 光谱分支：LR-HSI上采样 + HDRTB特征提取 -> F_spe
          3. 空间分支：HR-MSI + HDRTB特征提取 -> F_spa  
          4. 特征融合：F_spe + F_spa -> conv_fuse -> YFD
          5. 最终重建：YFD -> final_blk(HDRTB) -> Conv3x3 -> HR-HSI输出
        """
        # ---- Step 1: 准备跨模态辅助信息 ----
        # 将HRMSI下采样到与LRHSI相同的空间分辨率，作为光谱分支的辅助输入
        # 输入: hrmsi=[B, 4, 256, 256], scale_factor=0.25 → 输出: [B, 4, 64, 64]
        lrmsi = torch.nn.functional.interpolate(hrmsi, scale_factor=0.25, mode='bicubic')
        # 将LRHSI上采样到与HRMSI相同的空间分辨率，作为空间分支的辅助输入
        # 输入: lrhsi=[B, 172, 64, 64], scale_factor=4 → 输出: [B, 172, 256, 256]
        hrhsi =  torch.nn.functional.interpolate(lrhsi, scale_factor=4, mode='bilinear')

        # ==================== 光谱分支 (Spectral Branch) ====================
        # 拼接LR-HSI(172ch)和下采样的LR-MSI(4ch)作为光谱分支输入
        # ┌──────────────────────────────────────────────────────────────────────┐
        # │ Step1: cat((lrhsi=[B,172,64,64], lrmsi=[B,4,64,64]), dim=1)          │
        # │       → [B, 176, 64, 64]                                             │
        # │ Step2: hsiconv1=Conv2d(176→160, k=3, p=1)                            │
        # │       → [B, 160, 64, 64]                                              │
        # │ Step3: lrelu=LeakyReLU(0.2) 形状不变                                 │
        # │       → [B, 160, 64, 64]                                              │
        # └──────────────────────────────────────────────────────────────────────┘
        lrhsi = self.lrelu(self.hsiconv1(torch.cat((lrhsi, lrmsi), 1)))  # → [B, 160, 64, 64]

        # 首次上采样 ×2: 64×64 → 128×128
        # up=Upsample(scale_factor=2)，双线性插值上采样，通道数不变
        # 输入: [B, 160, 64, 64] → 输出: [B, 160, 128, 128]
        lrhsi = self.up(lrhsi)  # → [B, 160, 128, 128]

        # 记录当前空间尺寸(H,W)供后续patch_embed/patch_unembed使用
        # 输入: lrhsi.shape=(B, 160, 128, 128) → 输出: (128, 128)
        x_size = (lrhsi.shape[2], lrhsi.shape[3])  # → (128, 128)

        # 深拷贝上采样后的特征，用于后续残差连接（skip connection）
        # 输入: [B, 160, 128, 128] → 输出: [B, 160, 128, 128]
        lrhsi2 = lrhsi.clone()  # → [B, 160, 128, 128]

        # PatchEmbed: [B,C,H,W] → flatten(2) → transpose(1,2) → [B, H*W, C]
        # 将图像格式的特征展平为序列格式，供Transformer/HDRTB处理
        # 输入: [B, 160, 128, 128] → 输出: [B, 16384, 160] （128*128=16384个patch）
        lrhsi = self.patch_embed(lrhsi)  # → [B, 16384, 160]

        # 通过2个HDRTB(SwinBasedFeatFusionBlock)进行分层多尺度光谱特征提取
        # 每个HDRTB内部含4个HierarchicalTransformerBlock(SSCL)，使用渐进窗口
        # 输入: [B, 16384, 160] → 输出: [B, 16384, 160] （形状不变，语义增强）
        for ii,layer in enumerate(self.hsifeat):
            lrhsi = layer(lrhsi, x_size)  # → [B, 16384, 160]

        # PatchUnEmbed: [B,H*W,C] → transpose(1,2) → view(B,C,H,W) → [B,C,H,W]
        # 将序列格式转回图像格式
        # 输入: [B, 16384, 160], x_size=(128,128) → 输出: [B, 160, 128, 128]
        lrhsi = self.patch_unembed(lrhsi, x_size)  # → [B, 160, 128, 128]
        
        # 残差连接：HDRTB提取的特征 + 上采样后的初始特征（缓解梯度消失）
        # 输入: lrhsi=[B, 160, 128, 128], lrhsi2=[B, 160, 128, 128]
        # → 输出: [B, 160, 128, 128]
        lrhsi = lrhsi + lrhsi2  # → [B, 160, 128, 128]

        # 第二次上采样 ×2: 128×128 → 256×256（达到目标HR分辨率）
        # 输入: [B, 160, 128, 128] → 输出: [B, 160, 256, 256]
        lrhsi = self.up(lrhsi)  # → [B, 160, 256, 256]

        # LeakyReLU激活
        # 输入: [B, 160, 256, 256] → 输出: [B, 160, 256, 256]
        lrhsi = self.lrelu(lrhsi)  # → [B, 160, 256, 256]

        # 光谱分支末层卷积：通道从nf*2=160降到nf=80，与空间分支对齐
        # hsiconvlast=Conv2d(160→80, k=3, s=1, p=1, groups=4)
        # 输入: [B, 160, 256, 256] → 输出: [B, 80, 256, 256]
        lrhsi = self.hsiconvlast(lrhsi)  # → [B, 80, 256, 256]  ← F_spe

        # ==================== 空间分支 (Spatial Branch) ====================
        # 拼接HR-MSI(4ch)和上采样的HR-HSI(172ch)作为空间分支输入
        # ┌──────────────────────────────────────────────────────────────────────┐
        # │ Step1: cat((hrmsi=[B,4,256,256], hrhsi=[B,172,256,256]), dim=1)      │
        # │       → [B, 176, 256, 256]                                           │
        # │ Step2: msiconv1=Conv2d(176→40, k=3, p=1)  (nf//2=80//2=40)           │
        # │       → [B, 40, 256, 256]                                             │
        # │ Step3: lrelu=LeakyReLU(0.2) 形状不变                                  │
        # │       → [B, 40, 256, 256]                                             │
        # └──────────────────────────────────────────────────────────────────────┘
        hrmsi = self.lrelu(self.msiconv1(torch.cat((hrmsi, hrhsi), 1)))  # → [B, 40, 256, 256]

        # 记录当前空间尺寸(H,W)供后续patch_embed/patch_unembed使用
        # 输入: hrmsi.shape=(B, 40, 256, 256) → 输出: (256, 256)
        x_size = (hrmsi.shape[2], hrmsi.shape[3])  # → (256, 256)

        # 深拷贝空间分支初始特征，用于后续残差连接
        # 输入: [B, 40, 256, 256] → 输出: [B, 40, 256, 256]
        hrmsi2=hrmsi.clone()  # → [B, 40, 256, 256]

        # PatchEmbed: [B,C,H,W] → [B, H*W, C]
        # 输入: [B, 40, 256, 256] → 输出: [B, 65536, 40] （256*256=65536个patch）
        hrmsi = self.patch_embed(hrmsi)  # → [B, 65536, 40]

        # 通过2个HDRTB进行分层多尺度空间特征提取
        # dim=nf//2=40，input_resolution=(4,4)表示在256×256图像上的等效窗口
        # 输入: [B, 65536, 40] → 输出: [B, 65536, 40]
        for ii,layer in enumerate(self.msifeat):
            hrmsi = layer(hrmsi, x_size)  # → [B, 65536, 40]
        
        # 残差连接：空间分支初始特征 + HDRTB输出（先unembed再相加）
        # patch_unembed: [B, 65536, 40], x_size=(256,256) → [B, 40, 256, 256]
        # 再与hrmsi2=[B, 40, 256, 256]相加
        # → 输出: [B, 40, 256, 256]
        hrmsi = hrmsi2+ self.patch_unembed(hrmsi, x_size)  # → [B, 40, 256, 256]

        # LeakyReLU激活
        # 输入: [B, 40, 256, 256] → 输出: [B, 40, 256, 256]
        hrmsi = self.lrelu(hrmsi)  # → [B, 40, 256, 256]

        # 空间分支末层卷积：通道从nf//2=40提升到nf=80，与光谱分支对齐
        # msiconvlast=Conv2d(40→80, k=3, s=1, p=1)
        # 输入: [B, 40, 256, 256] → 输出: [B, 80, 256, 256]
        hrmsi = self.msiconvlast(hrmsi)  # → [B, 80, 256, 256]  ← F_spa

        # ==================== 特征融合 (Feature Fusion) ====================
        # 双分支特征逐元素相加（论文公式1: F_spe + F_spa），然后通过融合卷积层
        # ┌──────────────────────────────────────────────────────────────────────┐
        # │ Step1: hrmsi + lrhsi                                                 │
        # │   [B, 80, 256, 256] + [B, 80, 256, 256] → [B, 80, 256, 256]         │
        # │ Step2: conv_fuse = Sequential(                                       │
        # │   Conv2d(80→172, k=3, p=1, groups=4),  (分组卷积，172%4==0)         │
        # │   LeakyReLU(0.2))                                                    │
        # │   → [B, 172, 256, 256]                                               │
        # └──────────────────────────────────────────────────────────────────────┘
        yfd = self.conv_fuse(hrmsi + lrhsi)  # YFD → [B, 172, 256, 256]
        
        # ==================== 最终重建 (Final Reconstruction F_final) ====================
        # 通过最终HDRTB进行精细化重建和细节恢复
        # 记录当前空间尺寸
        # 输入: yfd.shape=(B, 172, 256, 256) → 输出: (256, 256)
        x_size = (yfd.shape[2], yfd.shape[3])  # → (256, 256)

        # PatchEmbed: 图像→序列
        # 输入: [B, 172, 256, 256] → 输出: [B, 65536, 172]
        yfd = self.patch_embed(yfd)  # → [B, 65536, 172]

        # 通过final_blk中的1个SwinBasedFeatFusionBlock_final_block进行精细化
        # dim=out_nc=172，使用Swin Transformer进行最终细节恢复
        # 输入: [B, 65536, 172] → 输出: [B, 65536, 172]
        for layer in self.final_blk:
            yfd = layer(yfd, x_size)  # → [B, 65536, 172]

        # PatchUnEmbed: 序列→图像
        # 输入: [B, 65536, 172], x_size=(256,256) → 输出: [B, 172, 256, 256]
        yfd = self.patch_unembed(yfd, x_size)  # → [B, 172, 256, 256]

        # 最终3×3卷积生成HR-HSI输出（无偏置，纯映射）
        # last=Conv2d(172→172, k=3, s=1, p=1, bias=False)
        # 输入: [B, 172, 256, 256] → 输出: [B, 172, 256, 256]
        co = self.last(yfd)  # → [B, 172, 256, 256]  ← 最终HR-HSI重建结果
        
        return co  # → [B, 172, 256, 256]
    

class HyDCFN(nn.Module):
    """Hyperspectral Image Fusion Network (HyDCFN) - HSSDCT顶层包装器
    
    【论文对应】论文整体框架的顶层封装，负责：
      1. 管理YDCFN解码器（核心双分支融合网络）
      2. 训练时对LR-HSI注入AWGN噪声以增强鲁棒性（可选，由snr参数控制）
    
    Main module that combines Swin Transformer-based feature extraction
    with deep convolutional networks for hyperspectral and multispectral
    image fusion.
    """

    def __init__(self, args):
        super(HyDCFN, self).__init__()
        self.snr = args.snr       # 信噪比(dB)，用于AWGN噪声注入。>0时启用噪声增强训练
        self.joint = args.network_mode  # 网络模式：0=single, 1=LRHSI+HRMSI(配对), 2=COCNN(三元组)
        
        # 初始化YDCFN解码器（HSSDCT的核心双分支网络）
        self.decoder = YDCFN(in_nc=args.bands, out_nc=args.bands, nf=args.nf, gc=args.gc, in_msi=args.msi_bands, groups=1, debug=args.DEBUG)
        print('Use the COCNN!')
        
    def awgn(self, x):
        """【AWGN噪声注入】Additive White Gaussian Noise（加性高斯白噪声）
        
        在训练阶段向LR-HSI添加高斯噪声以提升模型对噪声的鲁棒性。
        噪声功率由目标SNR决定：noise_power = signal_power / 10^(SNR/10)
        论文实验中默认SNR=35dB。
        """
        snr = 10**(self.snr/10.0)
        xpower = torch.sum(x**2)/x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape).cuda() * npower


    def forward(self,LRHSI, HRMSI, mode=0): ### Mode=0, default, mode=1: encode only, mode=2: decoded only
        """【HyDCFN前向传播】顶层前向传播接口
        
        Args:
            LRHSI: 低分辨率高光谱图像
            HRMSI: 高分辨率多光谱图像  
            mode: 运行模式
              - 0 (default): 正常模式，训练时可能注入AWGN噪声
              - 1: 仅编码模式
              - 2: 仅解码模式
        
        Returns:
            重建的高分辨率高光谱图像 (HR-HSI)
        """
        if self.snr>0 and mode==0 and self.joint==1:
            # 训练模式下且SNR>0时，向LR-HSI注入AWGN噪声以增强鲁棒性
            LRHSI = self.awgn(LRHSI)

        return self.decoder(LRHSI, HRMSI)
    


