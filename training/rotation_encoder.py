# -*- coding: utf-8 -*-
"""
旋转向量条件编码器。

设计说明：
- 输入的旋转向量为 [delta_azimuth, delta_elevation, delta_roll]（单位：度），
  角度量本质上是周期性的（0度和360度应视为同一朝向），若直接把原始角度值
  喂给 MLP，网络需要额外学习这种周期不连续性，容易在角度边界（如 359 -> 0）
  处产生跳变伪影。因此这里对每个分量分别取 sin/cos，将 3 维角度编码为 6 维
  连续特征，从根本上消除周期不连续问题（这是姿态/角度类条件编码的常见做法）。
- 编码后的 6 维向量经过一个两层 MLP 升维到 768（与 SD1.5 CLIP text encoder 的
  cross-attention 隐藏维度一致），再 reshape 成 (batch, 1, 768)，即产出一个
  长度为 1 的"伪文本 token"。训练时可以直接把它与 CLIP 输出的 (batch, 77, 768)
  在 token 维度拼接，得到 (batch, 78, 768) 作为 UNet 的 cross-attention 条件，
  不需要改动 UNet 的 cross_attention_dim。
"""
import torch
import torch.nn as nn


class RotationEncoder(nn.Module):
    def __init__(self, cross_attention_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cross_attention_dim),
        )

    def forward(self, rotation_vector: torch.Tensor) -> torch.Tensor:
        """
        rotation_vector: (batch, 3)，单位为角度（度），顺序为
                          [delta_azimuth, delta_elevation, delta_roll]
        返回: (batch, 1, cross_attention_dim)
        """
        # 角度 -> 弧度，再做 sin/cos 编码，得到 (batch, 6)
        radians = rotation_vector * (torch.pi / 180.0)
        sincos = torch.cat([torch.sin(radians), torch.cos(radians)], dim=-1)

        embedding = self.mlp(sincos)  # (batch, cross_attention_dim)
        return embedding.unsqueeze(1)  # (batch, 1, cross_attention_dim)

    def save_pretrained(self, path: str):
        torch.save(self.state_dict(), path)

    def load_pretrained(self, path: str, map_location=None):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        return self
