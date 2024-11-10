import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcls.registry import MODELS
from .attention import MultiheadPosAttention
from mmengine.model.weight_init import trunc_normal_
from ..utils import MultiheadAttention, resize_pos_embed, to_2tuple
from mmcv.cnn.bricks.transformer import FFN


class CFDLoss(nn.Module):

    def __init__(self,
                 name,
                 use_this,
                 alpha_cfd,
                 beta_cfd,
                 student_dims,
                 teacher_dims,
                 pos_dims,
                 ):
        super(CFDLoss, self).__init__()
        self.alpha_cfd = alpha_cfd
        self.beta_cfd = beta_cfd

        self.atten_arch = self.get_atten_arch(model_type='deit-s')

        self.proj_pos = nn.Sequential(nn.Conv2d(teacher_dims, teacher_dims, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(teacher_dims),
                                      nn.ReLU())

        self.proj_student = nn.Sequential(nn.Conv2d(student_dims, student_dims, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(student_dims),
                                      nn.ReLU())

        self.proj_teacher = nn.Sequential(nn.Conv2d(teacher_dims, teacher_dims, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(teacher_dims),
                                      nn.ReLU(),
                                      nn.Conv2d(teacher_dims, teacher_dims, 3, stride=1, padding=1)
                                      )

        self.pos_embed = nn.Parameter(torch.zeros(1, student_dims, 7, 7), requires_grad=True)
        self.pos_attention = MultiheadPosAttention(teacher_dims, num_heads=8, input_dims=student_dims, pos_dims=pos_dims)
        self.ffn = FFN(embed_dims=teacher_dims, feedforward_channels=384 * 4)

        # self.pos_attention = MultiheadPosAttention(teacher_dims, num_heads=atten_arch['num_heads'], input_dims=student_dims, pos_dims=pos_dims)
        # self.ffn = FFN(embed_dims=teacher_dims, feedforward_channels=atten_arch['feedforward_channels'])
        self.norm = nn.LayerNorm([teacher_dims])

        #self.attention = MultiheadAttention(teacher_dims, num_heads=8, input_dims=teacher_dims, v_shortcut=True)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

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

        assert H_s*2 == H_t and W_s*2 == W_t

        pos_emb.requires_grad = True
        preds_T.requires_grad = True

        dis_loss, corr_loss = self.get_dis_loss(preds_S, preds_T, pos_emb) 
        
        dis_loss = dis_loss * self.alpha_cfd
        corr_loss = corr_loss * self.beta_cfd
            
        return dis_loss, corr_loss


    def get_dis_loss(self, preds_S, preds_T, pos_emb):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        preds_S = self.proj_student(preds_S) + self.pos_embed.to(preds_S.device)
        pos_emb = self.proj_pos(pos_emb)
        pos_emb = torch.flatten(pos_emb.permute(0, 2, 3, 1), 1, 2)

        feat_T_corr = self.proj_teacher(preds_T)

        corr_loss_p = self.get_pixel_correlation_loss(F.normalize(feat_T_corr), F.normalize(preds_T))/N
        # corr_loss_a = loss_mse(torch.mean(F.normalize(feat_T_corr, dim=1)), torch.mean(F.normalize(preds_T, dim=1)))
        corr_loss_a = loss_mse(F.normalize(torch.mean(feat_T_corr,dim=1).contiguous().view(N,-1)), F.normalize(torch.mean(preds_T, dim=1).contiguous().view(N,-1)))
        corr_loss = corr_loss_p + 20 * corr_loss_a

        fea_S = self.pos_attention(torch.flatten(preds_S.permute(0, 2, 3, 1), 1, 2), pos_emb)
        fea_S = self.ffn(self.norm(fea_S))
        #fea_S = self.attention(fea_S)

        fea_S = F.normalize(fea_S, dim=2)
        fea_T = F.normalize(torch.flatten(preds_T.permute(0, 2, 3, 1), 1, 2), dim=2)
        feat_T_corr = F.normalize(torch.flatten(feat_T_corr.permute(0, 2, 3, 1), 1, 2), dim=2)

        dis_loss = (loss_mse(fea_S, feat_T_corr)) / N
        #loss_mse(fea_S, fea_T)

        return dis_loss, corr_loss

    def get_pixel_correlation_loss(self, preds, targets):
        loss_mse = nn.MSELoss(reduction='sum')
        device = preds.device
        BS, C, H, W = targets.shape

        preds = preds.permute(0,2,3,1).contiguous().view(BS, H*W, C)
        targets = targets.permute(0,2,3,1).contiguous().view(BS, H*W, C)

        mask = torch.eye(H*W).bool().to(device)
        mask = mask.repeat(BS, 1, 1)

        preds_matrix = torch.cdist(preds, preds, p=2)[~mask].contiguous().view(BS*H*W, -1)
        targets_matrix = torch.cdist(targets, targets, p=2)[~mask].contiguous().view(BS*H*W, -1)
        # preds_matrix = torch.cdist(preds, targets, p=2).contiguous().view(BS*H*W, -1)
        # targets_matrix = torch.cdist(targets, targets, p=2).contiguous().view(BS*H*W, -1)

        loss = loss_mse(preds_matrix, targets_matrix)

        return loss
    
    def get_atten_arch(self, model_type):
        if model_type not in ['deit-t', 'deit-s', 'deit-b']:
            raise NotImplmentedError('The model type ' + model_type, 'has not been implemented yet')
        
        atten_arch_zoo = {
            'deit-t': {'num_heads':3, 'feedforward_channels':192 * 4},
            'deit-s': {'num_heads':6, 'feedforward_channels':384 * 4},
            'deit-b': {'num_heads':12, 'feedforward_channels':768 * 4}
        }

        return atten_arch_zoo[model_type]

class MFDLoss(nn.Module):

    def __init__(self,
                 name,
                 use_this,
                 alpha_mfd,
                 student_dims,
                 teacher_dims,
                 pos_dims,
                 ):
        super(MFDLoss, self).__init__()
        self.alpha_mfd = alpha_mfd

        # self.atten_arch = self.get_atten_arch(model_type='deit-s')

        self.projector_0 = AttentionProjector(student_dims=student_dims[0], 
                                              teacher_dims=teacher_dims[0],
                                              pos_dims=pos_dims,
                                              student_resolution=(56,56))
        
        self.projector_1 = AttentionProjector(student_dims=student_dims[1], 
                                              teacher_dims=teacher_dims[1],
                                              pos_dims=pos_dims,
                                              student_resolution=(28,28))
        
        self.projector_2 = AttentionProjector(student_dims=student_dims[2], 
                                              teacher_dims=teacher_dims[2],
                                              pos_dims=pos_dims,
                                              student_resolution=(14,14))
        
        self.projector_3 = AttentionProjector(student_dims=student_dims[3], 
                                              teacher_dims=teacher_dims[3],
                                              pos_dims=pos_dims,
                                              student_resolution=(7,7))
        
        self.projector = [self.projector_0, self.projector_1, self.projector_2, self.projector_3]


    def forward(self,
                fea_s,
                fea_t,
                ):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        # H_s, W_s = preds_S.shape[-2:]
        # H_t, W_t = preds_T.shape[-2:]

        # assert H_s*2 == H_t and W_s*2 == W_t

        loss = 0
        for i in range(4):
            index = -1 - i
            preds_S = fea_s[index]
            preds_T = fea_t[index][0]

            if i == 3:
                pos_emb = fea_t[index-1]
            else:
                pos_emb = fea_t[index-1][0]

            pos_emb.requires_grad = True
            loss += self.get_dis_loss(preds_S, preds_T, pos_emb, index) 
            
        loss = loss * self.alpha_mfd / 4.

        return loss


    def get_dis_loss(self, preds_S, preds_T, pos_emb, index):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        fea_S = self.projector[index](preds_S, pos_emb)

        fea_S = F.normalize(fea_S, dim=2)
        fea_T = F.normalize(torch.flatten(preds_T.permute(0, 2, 3, 1), 1, 2), dim=2)

        dis_loss = loss_mse(fea_S, fea_T) / N

        return dis_loss
    



class AttentionProjector(nn.Module):
    def __init__(self,
                 student_dims,
                 teacher_dims,
                 pos_dims,
                 student_resolution,
                 ):
        super(AttentionProjector, self).__init__()
        self.proj_pos = nn.Sequential(nn.Conv2d(teacher_dims, teacher_dims, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(teacher_dims),
                                      nn.ReLU())

        self.proj_student = nn.Sequential(nn.Conv2d(student_dims, student_dims, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(student_dims),
                                      nn.ReLU())

        self.pos_embed = nn.Parameter(torch.zeros(1, student_dims, student_resolution[0], student_resolution[1]), requires_grad=True)
        self.pos_attention = MultiheadPosAttention(teacher_dims, num_heads=8, input_dims=student_dims, pos_dims=pos_dims)
        self.ffn = FFN(embed_dims=teacher_dims, feedforward_channels=384 * 4)

        self.norm = nn.LayerNorm([teacher_dims])

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self,
                preds_S,
                pos_emb,
                ):
        
        preds_S = self.proj_student(preds_S) + self.pos_embed.to(preds_S.device)
        pos_emb = self.proj_pos(pos_emb)
        pos_emb = torch.flatten(pos_emb.permute(0, 2, 3, 1), 1, 2)

        fea_S = self.pos_attention(torch.flatten(preds_S.permute(0, 2, 3, 1), 1, 2), pos_emb)
        fea_S = self.ffn(self.norm(fea_S))

        return fea_S

class FDLoss(nn.Module):

    def __init__(self,
                 name,
                 use_this,
                 alpha_fd,
                 student_dims,
                 teacher_dims,
                 pos_dims,
                 ):
        super(FDLoss, self).__init__()
        self.alpha_fd = alpha_fd

        self.atten_arch = self.get_atten_arch(model_type='deit-s')


        self.proj_pos = nn.Sequential(nn.Conv2d(teacher_dims, teacher_dims, 1, stride=1),
                                      nn.BatchNorm2d(teacher_dims),
                                      nn.ReLU())

        self.proj_student = nn.Sequential(nn.Conv2d(student_dims, teacher_dims, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(teacher_dims),
                                      nn.ReLU())

        self.student_self_attention = MultiheadAttention(teacher_dims, num_heads=6, input_dims=teacher_dims, v_shortcut=False)
        self.shared_self_attention = MultiheadAttention(teacher_dims, num_heads=6, input_dims=teacher_dims, v_shortcut=False)

        self.query = nn.Embedding(196, 384)
        self.pos_embed_student = nn.Parameter(torch.zeros(1, student_dims, 7, 7), requires_grad=True)
        self.pos_embed_teacher = nn.Parameter(torch.zeros(1, teacher_dims, 14, 14), requires_grad=True)
        self.pos_attention = MultiheadPosAttention(teacher_dims, num_heads=6, input_dims=teacher_dims, pos_dims=pos_dims)
        self.ffn = FFN(embed_dims=teacher_dims, feedforward_channels=384 * 4)   

        # self.pos_attention = MultiheadPosAttention(teacher_dims, num_heads=atten_arch['num_heads'], input_dims=student_dims, pos_dims=pos_dims)
        # self.ffn = FFN(embed_dims=teacher_dims, feedforward_channels=atten_arch['feedforward_channels'])
        self.norm = nn.LayerNorm([teacher_dims])

        #self.attention = MultiheadAttention(teacher_dims, num_heads=8, input_dims=teacher_dims, v_shortcut=True)

        if self.pos_embed_student is not None:
            trunc_normal_(self.pos_embed_student, std=0.02)
        if self.pos_embed_teacher is not None:
            trunc_normal_(self.pos_embed_teacher, std=0.02)

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


        assert H_s*2 == H_t and W_s*2 == W_t

        pos_emb.requires_grad = True
        loss = self.get_dis_loss(preds_S, preds_T, pos_emb) * self.alpha_fd

        return loss


    def get_dis_loss(self, preds_S, preds_T, pos_emb):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        pos_emb = (self.query.weight.transpose(0,1).contiguous().expand(N, C, H*W).contiguous().view(N,C,H,W) + self.pos_embed_teacher).to(preds_S.device)
        # pos_emb = self.pos_embed_teacher.to(preds_S.device).expand(N,C,H,W)

        preds_S = preds_S + self.pos_embed_student.to(preds_S.device).contiguous()
        preds_S = self.proj_student(preds_S)
        pos_emb = self.proj_pos(pos_emb.contiguous())

        fea_S = torch.flatten(preds_S.permute(0, 2, 3, 1), 1, 2).contiguous()
        pos_emb = torch.flatten(pos_emb.permute(0, 2, 3, 1), 1, 2).contiguous()

        fea_S = self.student_self_attention(fea_S) + fea_S
        fea_S = self.shared_self_attention(fea_S) + fea_S
        pos_emb = self.shared_self_attention(pos_emb)

        fea_S = self.pos_attention(fea_S, pos_emb)
        fea_S = self.ffn(self.norm(fea_S))

        fea_S = F.normalize(fea_S, dim=2)
        fea_T = F.normalize(torch.flatten(preds_T.permute(0, 2, 3, 1), 1, 2), dim=2)

        dis_loss = loss_mse(fea_S, fea_T) / N

        return dis_loss
    
    def get_atten_arch(self, model_type):
        if model_type not in ['deit-t', 'deit-s', 'deit-b']:
            raise NotImplmentedError('The model type ' + model_type, 'has not been implemented yet')
        
        atten_arch_zoo = {
            'deit-t': {'num_heads':3, 'feedforward_channels':192 * 4},
            'deit-s': {'num_heads':6, 'feedforward_channels':384 * 4},
            'deit-b': {'num_heads':12, 'feedforward_channels':768 * 4}
        }

        return atten_arch_zoo[model_type]



class ViTKDLoss(nn.Module):

    """PyTorch version of `ViTKD: Practical Guidelines for ViT feature knowledge distillation` """

    def __init__(self,
                 name,
                 use_this,
                 student_dims,
                 teacher_dims,
                 alpha_vitkd=0.00003,
                 beta_vitkd=0.000003,
                 lambda_vitkd=0.5,
                 ):
        super(ViTKDLoss, self).__init__()
        self.alpha_vitkd = alpha_vitkd
        self.beta_vitkd = beta_vitkd
        self.lambda_vitkd = lambda_vitkd
    
        if student_dims != teacher_dims:
            self.align2 = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(2)])
            self.align = nn.Linear(student_dims, teacher_dims, bias=True)
        else:
            self.align2 = None
            self.align = None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))

        self.generation = nn.Sequential(
                nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), 
                nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1))

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(List): [B*2*N*D, B*N*D], student's feature map
            preds_T(List): [B*2*N*D, B*N*D], teacher's feature map
        """
        low_s = preds_S[0]
        low_t = preds_T[0]
        high_s = preds_S[1]
        high_t = preds_T[1]

        B = low_s.shape[0]
        loss_mse = nn.MSELoss(reduction='sum')

        '''ViTKD: Mimicking'''
        if self.align2 is not None:
            for i in range(2):
                if i == 0:
                    xc = self.align2[i](low_s[:,i]).unsqueeze(1)
                else:
                    xc = torch.cat((xc, self.align2[i](low_s[:,i]).unsqueeze(1)),dim=1)
        else:
            xc = low_s

        loss_lr = loss_mse(xc, low_t) / B * self.alpha_vitkd

        '''ViTKD: Generation'''
        if self.align is not None:
            x = self.align(high_s)
        else:
            x = high_s

        # Mask tokens
        B, N, D = x.shape
        x, mat, ids, ids_masked = self.random_masking(x, self.lambda_vitkd)
        mask_tokens = self.mask_token.repeat(B, N - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))
        mask = mat.unsqueeze(-1)

        hw = int(N**0.5)
        x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)
        x = self.generation(x).flatten(2).transpose(1,2)

        loss_gen = loss_mse(torch.mul(x, mask), torch.mul(high_t, mask))
        loss_gen = loss_gen / B * self.beta_vitkd / self.lambda_vitkd
            
        return loss_lr + loss_gen

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:L]

        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore, ids_masked

class MGDLoss(nn.Module):

    """PyTorch version of `Masked Generative Distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00007
        lambda_mgd (float, optional): masked ratio. Defaults to 0.5
    """
    def __init__(self,
                 name,
                 use_this,
                 student_channels,
                 teacher_channels,
                 alpha_mgd=0.00007,
                 lambda_mgd=0.15,
                 ):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
    
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))


    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)
    
        loss = self.get_dis_loss(preds_S, preds_T)*self.alpha_mgd
            
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,C,1,1)).to(device)
        # mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat < self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss



    