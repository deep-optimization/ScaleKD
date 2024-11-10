import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcls.registry import MODELS

from ..heads.vision_transformer_head import VisionTransformerClsHead
from ..losses import LabelSmoothLoss

@MODELS.register_module()
class TCDLoss(nn.Module):
        # """
        #     TCDLoss is short for "Transformer-CNN-Distillation-Loss".
        # """
    def __init__(self,
                 name,
                 use_this,
                 student_dims,
                 teacher_dims,
                 alpha_tcd,
                 beta_tcd,
                 lambda_tcd=0.5,
                 ):
        super(TCDLoss, self).__init__()
        self.alpha_tcd = alpha_tcd
        self.beta_tcd = beta_tcd
        self.lambda_tcd = lambda_tcd
    
        if student_dims != teacher_dims:
            self.align = nn.Conv2d(student_dims, teacher_dims, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generator = nn.Sequential(
            nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1))
        
        self.cls_head = None

        self.cls_loss = LabelSmoothLoss(label_smooth_val=0.1, mode='original')
        self.dis_loss = nn.MSELoss(reduction='none')


    def forward(self,
                preds_S,
                preds_T,
                gt,
                cls_token):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        
        BS, C, H_t, W_t = preds_T.shape
        H_s, W_s = preds_S.shape[-2:]
        device = preds_S.device

        assert H_s*2 == H_t and W_s*2 == W_t

        feat_S = self.get_upsampled_student_feat(preds_S)
        feat_T = preds_T

        binary_mask = torch.rand((BS,1,H_t,W_t)).to(device)
        binary_mask = torch.where(binary_mask < self.lambda_tcd, 0, 1).to(device)

        mixed_feat = torch.mul(feat_S, binary_mask) + torch.mul(feat_T, 1-binary_mask)

        mixed_feat = self.generator(mixed_feat)
        head_feat = mixed_feat.view(BS,C,-1).permute(0,2,1)
        logits = self.cls_head.layers.head(cls_token)

        cls_loss = self.alpha_tcd * self.cls_head._get_loss(logits, gt)['loss']
        dis_loss = self.beta_tcd * self.dis_loss(mixed_feat, feat_T).mul(binary_mask).sum()
            
        return cls_loss, dis_loss


    def get_upsampled_student_feat(self, preds_S):
        preds_S = F.interpolate(preds_S, scale_factor=2, mode='bilinear')
        
        if self.align is not None:
            preds_S = self.align(preds_S)
        
        return preds_S


    def set_head(self, cls_head):
        self.cls_head = cls_head

        return