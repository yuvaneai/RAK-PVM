import os

import torch
import torch.nn as nn
from functools import partial

from backbones.vits import create_vit, Block, VisionTransformer
from timm.models.vision_transformer import PatchEmbed

from transformers import AutoTokenizer, BertConfig, BertTokenizer, logging

logging.set_verbosity_error()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Patch_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
       """
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        return x

class Cls_Token(nn.Module):
    """Construct the embeddings from patch, position embeddings.
       """
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        # B = batchsize
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        return cls_tokens

class Pos_Embed(nn.Module):
    """Construct the embeddings from patch, position embeddings.
       """
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        # B = x.shape[0]
        x = x + self.pos_embed[:, :x.size(1), :]
        return x

class BlockF(nn.Module):
    """Construct the embeddings from patch, position embeddings.
       """
    def __init__(self, depth, embed_dim, num_heads, drop_path_rate, norm_layer, attn_drop=0.,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, vit_grad_ckpt=False, vit_ckpt_layer=0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_path_rate, attn_drop=attn_drop, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(
                        vit_grad_ckpt and i >= depth - vit_ckpt_layer)
            )
            for i in range(depth)])

    def forward(self, x, index, register_blk=-1):
        if index == 0:
            x = self.blocks[0](x, register_blk == register_blk)
        else:
            x = self.blocks[index](x, register_blk == index)

        return x

class Norm(nn.Module):
    """Construct the embeddings from patch, position embeddings.
       """
    def __init__(self, norm_layer, vision_width):
        super().__init__()
        self.norm = norm_layer(vision_width)

    def forward(self, x, register_blk=-1):
        x = self.norm(x)
        return x

class Vit_ImageEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "resnet_50",
                 hidden_dim: int = 2048,
                 output_dim: int = 128
                 ):
        super(Vit_ImageEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim

        if "vit" in model_name:
            # vit的参数设置
            vit_name = model_name[4:]
            image_size = 224
            patch_size = 16
            in_chans = 3
            drop_path_rate = 0.

            if vit_name == 'base':
                vision_width = 768
                self.feature_dim = vision_width
                depth = 12
                self.depth = depth
                num_heads = 12
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True)
                pretrain_state_dict = checkpoint["model"]

            norm_layer = partial(nn.LayerNorm, eps=1e-6)

            # Path Embedding
            self.patch_embeddings = Patch_Embeddings(
                img_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=vision_width)
            self.patch_embeddings.load_state_dict(pretrain_state_dict, strict=False)
            for param in self.patch_embeddings.parameters():
                param.requires_grad = False

            num_patches = self.patch_embeddings.patch_embed.num_patches

            # CLS_Token
            self.cls_token = Cls_Token(embed_dim=vision_width)
            self.cls_token.load_state_dict(pretrain_state_dict, strict=False)
            for param in self.cls_token.parameters():
                param.requires_grad = False

            # Position Embedding
            self.pos_embed = Pos_Embed(embed_dim=vision_width, num_patches = num_patches)
            self.pos_embed.load_state_dict(pretrain_state_dict, strict=False)
            for param in self.pos_embed.parameters():
                param.requires_grad = False
            self.pos_drop = nn.Dropout(p=drop_path_rate)

            # Block
            self.blocks = BlockF(depth=depth, embed_dim=vision_width, num_heads=num_heads, drop_path_rate=drop_path_rate, norm_layer=norm_layer)
            self.blocks.load_state_dict(pretrain_state_dict, strict=False)
            for param in self.blocks.parameters():
                param.requires_grad = False

            # Normalization
            self.norm = Norm(norm_layer=norm_layer, vision_width=vision_width)
            self.norm.load_state_dict(pretrain_state_dict, strict=False)
            for param in self.norm.parameters():
                param.requires_grad = False

            self.image_output_proj = nn.Linear(vision_width, self.output_dim)

        else:
            print("请输入vit模型")


    def forward(self, x):

        """
        x: [B, C, H, W]
        returns: cls_token [B, 768], patch_tokens [B, num_patches, 768]
        """

        # Patch embedding
        x = self.patch_embeddings(x) # [B, num_patches, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token(x)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, embed_dim]

        # Add position embedding
        x = x + self.pos_embed(x)

        # Transformer encoder
        for i in range(0, self.depth):
            x = self.blocks(x, index=i, register_blk=i)

        x = self.norm(x)

        return x[:, 0].contiguous(), x[:, 1:].contiguous()





