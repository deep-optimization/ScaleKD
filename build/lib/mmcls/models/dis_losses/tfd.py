import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcls.registry import MODELS
from .attention import MultiheadPosAttention
from mmengine.model.weight_init import trunc_normal_
from ..utils import MultiheadAttention, resize_pos_embed, to_2tuple
from mmcv.cnn.bricks.transformer import FFN, PatchMerging


@MODELS.register_module()
class TFDLoss(nn.Module):

    def __init__(self,
                 name,
                 use_this,
                 alpha_tfd,
                 student_dims,
                 teacher_dims,
                 pos_dims,
                 ):
        super(TFDLoss, self).__init__()
        self.alpha_tfd = alpha_tfd

        # self.merge_patch = PatchMerging(in_channels=teacher_dims//2,
        #                                 out_channels=teacher_dims,
        #                                 kernel_size=(2,2),
        #                                 stride=2,)

        # self.query = nn.Parameter(torch.zeros(1, teacher_dims, 7, 7), requires_grad=True)
        self.query = nn.Embedding(49, teacher_dims)

        self.proj_pos = nn.Sequential(nn.Conv2d(teacher_dims, teacher_dims, 1),
                                      nn.ReLU(),
                                      )

        self.proj_student = nn.Sequential(nn.Conv2d(student_dims, student_dims, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(student_dims),
                                      nn.ReLU())

        self.pos_embed = nn.Parameter(torch.zeros(1, student_dims, 7, 7), requires_grad=True)
        self.pos_attention = MultiheadPosAttention(teacher_dims, num_heads=8, input_dims=student_dims, pos_dims=pos_dims)
        self.ffn = FFN(embed_dims=teacher_dims, feedforward_channels=teacher_dims * 4)

        self.norm = nn.LayerNorm([teacher_dims])


        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
            # trunc_normal_(self.query, std=0.02)

    def forward(self,
                preds_S,
                preds_T,
                pos_emb):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        H_s, W_s = preds_S.shape[-2:]
        H_t, W_t = preds_T.shape[-2:]

        assert H_s == H_t and W_s == W_t

        N, C, H, W = pos_emb.shape

        # pos_emb.requires_grad = True
        # pos_emb = pos_emb.permute(0,2,3,1).contiguous().view(N,-1,C)
        # pos_emb, out_size = self.merge_patch(pos_emb, (14,14))
        # pos_emb = pos_emb.permute(0,2,1).contiguous().view(N,C*2,H//2,W//2)
        pos_emb = self.query.weight.view(1,7,7,C*2).permute(0,3,1,2).repeat(N,1,1,1)
        # pos_emb = self.query.repeat(N,1,1,1)
        loss = self.get_dis_loss(preds_S, preds_T, pos_emb) * self.alpha_tfd
            
        return loss


    def get_dis_loss(self, preds_S, preds_T, pos_emb):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        preds_S = self.proj_student(preds_S) + self.pos_embed.to(preds_S.device)
        pos_emb = self.proj_pos(pos_emb)
        pos_emb = torch.flatten(pos_emb.permute(0, 2, 3, 1), 1, 2)

        fea_S = self.pos_attention(torch.flatten(preds_S.permute(0, 2, 3, 1), 1, 2), pos_emb)
        fea_S = self.ffn(self.norm(fea_S))


        fea_S = F.normalize(fea_S, dim=2)
        fea_T = F.normalize(torch.flatten(preds_T.permute(0, 2, 3, 1), 1, 2), dim=2)

        dis_loss = loss_mse(fea_S, fea_T) / N

        return dis_loss
    


