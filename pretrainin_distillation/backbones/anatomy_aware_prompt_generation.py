
from typing import List, Tuple, Type, Optional

from backbones.two_way_transformer import TwoWayTransformer
from backbones.common import LayerNorm2d

import torch
import torch.nn as nn


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        dtype = self.positional_encoding_gaussian_matrix.dtype

        y = torch.linspace(0.5 / h, 1 - 0.5 / h, h, device=device, dtype=dtype)
        x = torch.linspace(0.5 / w, 1 - 0.5 / w, w, device=device, dtype=dtype)

        y_embed, x_embed = torch.meshgrid(y, x, indexing='ij')

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

class EfficientHyperNetworks(nn.Module):

    def __init__(self, transformer_dim: int = 128, num_mask_tokens: int = 4, output_dim: int = 16):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_mask_tokens = num_mask_tokens
        self.output_dim = output_dim

        self.shared_base = nn.Sequential(
            nn.Linear(transformer_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.specific_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, output_dim),
                nn.Tanh()
            ) for _ in range(num_mask_tokens)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, mask_tokens_out: torch.Tensor) -> torch.Tensor:
        batch_size = mask_tokens_out.shape[0]

        flat_input = mask_tokens_out.reshape(-1, self.transformer_dim)

        base_features = self.shared_base(flat_input)

        head_outputs = []
        for i in range(self.num_mask_tokens):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            token_features = base_features[start_idx:end_idx]

            head_output = self.specific_heads[i](token_features)
            head_outputs.append(head_output)

        hyper_in = torch.stack(head_outputs, dim=1)

        return hyper_in

class Seg_Prompt_Learner(nn.Module):
    def __init__(self,
                 *,
                 transformer_dim: int,
                 num_multimask_outputs: int = 3,
                 activation: Type[nn.Module] = nn.GELU,
                 ):
        super(Seg_Prompt_Learner, self).__init__()

        self.transformer_dim = transformer_dim

        self.pe_layer = PositionEmbeddingRandom(transformer_dim // 2)
        self.img_proj = nn.Linear(768, self.transformer_dim, bias=True)

        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(
            depth=1,
            embedding_dim=transformer_dim,
            mlp_dim=512,
            num_heads=4,
        )

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, 64, kernel_size=2, stride=2),
            LayerNorm2d(64),
            activation(),

            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            activation(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            activation(),
        )

        self.output_hypernetworks_mlps = EfficientHyperNetworks(transformer_dim=transformer_dim, num_mask_tokens=self.num_mask_tokens, output_dim=16)


    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]

        # Prepare output
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        B_img = image_embeddings.shape[0]
        B_prompt = tokens.shape[0]
        repeat_factor = B_prompt // B_img
        if repeat_factor > 1:
            image_embeddings = torch.repeat_interleave(image_embeddings, repeat_factor, dim=0)
            image_pe = torch.repeat_interleave(image_pe, repeat_factor, dim=0)
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in =  self.output_hypernetworks_mlps(mask_tokens_out)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return masks






