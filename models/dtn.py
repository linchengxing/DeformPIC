# This file is modified by SiT
# https://github.com/willisma/SiT/blob/main/models.py
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Mlp
from pytorch3d.ops import sample_farthest_points, knn_points


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Group_without_norm(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        center, _ = sample_farthest_points(xyz, K=self.num_group) # [B, npoint, 3]  [B, npoint]
        _, idx, _ = knn_points(center, xyz, K=self.group_size, return_nn=False) # [B, npoint, k]

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        return neighborhood, center
    
class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )   

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

##################################################
#               Embedding Layers                 #
##################################################
    
class PatchEmbedder(nn.Module):
    """
    Patch embedding: bridge encoder and diffusion transformer
    """
    def __init__(self, hidden_size, num_patches, patch_size):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.proj = Encoder(hidden_size)

    def forward(self, inputs):
        """
        inputs: (B, N, C) tensor of point clouds
        Returns:
        embeddings: (B, num_patches, hidden_size) tensor of patch embeddings
        """
        embeddings = self.proj(inputs)
        return embeddings

class ExampleEmbedder(nn.Module):
    """
    Embeds example infomation into vector representations.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.proj = nn.Linear(input_size, output_size, bias=False)
        
    def forward(self, incontext):
        return self.proj(incontext)


#################################################################################
#                                 Core DTN Model                                #
#################################################################################

class DTNBlock(nn.Module):
    """
    A DTN block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # self-attn with adaLN-Zero conditioning:
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Conv1d(hidden_size, out_channels, 1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x.transpose(-1, -2)).transpose(-1, -2)
        return x
    

class TransferNetwork(nn.Module):
    def __init__(
        self,
        in_channels=3,
        input_size=1024,
        num_patches=64,
        patch_size=32,
        y_dim=384,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.input_channels = in_channels
        self.input_size = input_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbedder(hidden_size, num_patches, patch_size)
        self.y_embedder = ExampleEmbedder(y_dim, hidden_size)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DTNBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size * in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.first_conv[0].weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.first_conv[1].weight, 1)
        nn.init.constant_(self.x_embedder.proj.first_conv[1].bias, 0)
        w = self.x_embedder.proj.first_conv[3].weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        w = self.x_embedder.proj.second_conv[0].weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.second_conv[1].weight, 1)
        nn.init.constant_(self.x_embedder.proj.second_conv[1].bias, 0)
        w = self.x_embedder.proj.second_conv[3].weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize label embedding table:
        w = self.y_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def depatchify(self, x):
        """
        Converts the output of the final layer from patches to a full point cloud.
        x: (B, num_patches, patch_size * in_channels) tensor
        Returns:
        x: (B, *input_size, in_channels) tensor
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.x_embedder.num_patches, self.x_embedder.patch_size, self.input_channels)
        return x

    def forward(self, x, y):
        """
        Forward pass of SiT.
        x: (N, G, S, C) tensor of spatial inputs (raw point clouds patches)
        t: (N,) tensor of diffusion timesteps
        y: (N, C) tensor of in-context example infos
        """
        x = self.x_embedder(x) + self.pos_embed
        c = self.y_embedder(y)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.depatchify(x)
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, seq_len, dtype=np.float32):
    """
    Generate 1D sine-cosine positional embeddings.

    Args:
        embed_dim (int): Dimension of the positional embedding (must be even).
        seq_len (int): Length of the sequence.
        dtype: Data type of the output (default: np.float32).

    Returns:
        np.ndarray: Positional embeddings of shape (seq_len, embed_dim).
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even"

    position = np.arange(seq_len, dtype=dtype)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, embed_dim, 2, dtype=dtype) * -(np.log(10000.0) / embed_dim))  # (embed_dim/2,)

    pos_embed = np.zeros((seq_len, embed_dim), dtype=dtype)
    pos_embed[:, 0::2] = np.sin(position * div_term)
    pos_embed[:, 1::2] = np.cos(position * div_term)

    return pos_embed
