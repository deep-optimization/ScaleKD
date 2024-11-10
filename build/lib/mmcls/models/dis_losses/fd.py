import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcls.registry import MODELS

@MODELS.register_module()
class FDLoss(nn.Module):

    def __init__(self,
                 name,
                 use_this,
                 alpha_fd,
                 student_dims,
                 teacher_dims,
                 ):
        super(FDLoss, self).__init__()
        self.alpha_fd = alpha_fd
    
        if student_dims != teacher_dims:
            self.align = nn.Conv2d(student_dims, teacher_dims, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1))



    def forward(self,
                preds_S,
                preds_T,
                ):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        H_s, W_s = preds_S.shape[-2:]
        H_t, W_t = preds_T.shape[-2:]


        assert H_s == H_t and W_s == W_t

        if self.align is not None:
            preds_S = self.align(preds_S)
    
        loss = self.get_dis_loss(preds_S, preds_T)*self.alpha_fd
            
        return loss


    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        new_fea = self.generation(preds_S)

        new_fea = F.normalize(new_fea, dim=1, p=2)
        preds_T = F.normalize(preds_T, dim=1, p=2)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss