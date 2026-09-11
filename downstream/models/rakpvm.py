import os
import torch.nn.functional as F

from backbones.anatomy_patch_relevant import DensePromptForTest, AnatomyStudent
from backbones.anatomy_aware_prompt_generation import Seg_Prompt_Learner
from backbones.image_encoder import Vit_ImageEncoder

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class RAK_PVM(nn.Module):
    def __init__(self, modelname: str, num_classes: int, learn_type, pretrained: bool = True,
                 prompt_tokens: int = 5, prompt_dropout: float = 0.0, prompt_type: str = 'shallow'):
        super().__init__()
        self.learn_type = learn_type
        print("learn_type:", self.learn_type)

        self.dense_prompt_test = DensePromptForTest(output_dim=256)

        self.prompt_learner = Seg_Prompt_Learner(transformer_dim=256)

        self.img_encoder_q = Vit_ImageEncoder(
            model_name="vit_base", output_dim=256)

        self.anatomy_box_S = AnatomyStudent(
            patch_dim=256,
            hidden_dim=128,
            num_queries=30
        )
        pretrain_ckpt = os.path.join(BASE_DIR, f"../pretraining+distillation/result/pretrain/ckpts/example.ckpt")
        distill_ckpt = os.path.join(BASE_DIR, f"../pretraining+distillation/result/distill/ckpts/example.ckpt")
        our_state_dict = torch.load(pretrain_ckpt, map_location=torch.device("cpu"))
        our_state_dict_s = torch.load(distill_ckpt, map_location=torch.device("cpu"))

        our_state_dict = our_state_dict["state_dict"]
        our_state_dict_s = our_state_dict_s["state_dict"]


        img_encoder_state_dict = {k.replace("img_encoder_q.", ""): v  for k, v in our_state_dict.items() if k.startswith("img_encoder_q.")}
        anatomy_box_S_state_dict = {k.replace("anatomy_box_S.", ""): v  for k, v in our_state_dict_s.items() if k.startswith("anatomy_box_S.")}
        prompt_learner_state_dict = {k.replace("prompt_learner.", ""): v for k, v in our_state_dict.items() if k.startswith("prompt_learner.")}
        dense_prompt_test_state_dict = {k.replace("dense_prompt_test.", ""): v for k, v in our_state_dict.items() if
                                     k.startswith("dense_prompt_test.")}

        self.prompt_learner.load_state_dict(prompt_learner_state_dict, strict=False)
        self.img_encoder_q.load_state_dict(img_encoder_state_dict, strict=False)
        self.dense_prompt_test.load_state_dict(dense_prompt_test_state_dict, strict=False)
        self.anatomy_box_S.load_state_dict(anatomy_box_S_state_dict, strict=False)

        for param in self.prompt_learner.parameters():
            param.requires_grad = False

        for param in self.img_encoder_q.parameters():
            param.requires_grad = False

        for param in self.dense_prompt_test.parameters():
            param.requires_grad = False

        for param in self.anatomy_box_S.parameters():
            param.requires_grad = False
        # classifier
        self.head = nn.Linear(768, num_classes)



    def forward_features(self, x, target_label, register_blk=-1, is_train=False):
        image = x
        image_embeds = self.img_encoder_q.patch_embeddings(image)
        image_embeds = self.prompt_learner.img_proj(image_embeds)
        B, HW, C = image_embeds.shape
        H = W = int(HW ** 0.5)
        image_embedding_size = (H, W)

        image_embeddings = image_embeds.permute(0, 2, 1).reshape(B, C, H, W)
        image_pe = self.prompt_learner.pe_layer(image_embedding_size)
        image_pe = image_pe.repeat(B, 1, 1, 1)

        student_out = self.anatomy_box_S(image_embeds)

        sparse_embeddings = student_out

        background_S = self.dense_prompt_test(batch_size=B)
        background_S = F.normalize(background_S, dim=-1)
        dense_embeddings = background_S.view(B, C, 1, 1).expand(B, C, H, W)

        low_res_masks = self.prompt_learner(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        prompt_rgb = torch.sigmoid(low_res_masks)
        fused_img = image * (1 + prompt_rgb)
        fused_img = torch.clamp(fused_img, -1, 1)
        feats, _ = self.img_encoder_q(fused_img)

        return feats

    def forward(self, x, is_train):
        x = self.forward_features(x, is_train)
        x = self.head(x)
        return x