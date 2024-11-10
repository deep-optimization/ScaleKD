import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcls.registry import MODELS
from mmcv.cnn.bricks.transformer import FFN, PatchMerging

@MODELS.register_module()
class MultiStageFDLoss(nn.Module):

    def __init__(self,
                 name,
                 use_this,
                 alpha_mfd,
                 student_dims,
                 teacher_dims,
                 ):
        super(MultiStageFDLoss, self).__init__()
        self.alpha_mfd = alpha_mfd
    
        self.projector = FeatureProjector(student_dims, teacher_dims)

    def forward(self,
                preds_S,
                preds_T,
                ):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map

        """

        preds_S = self.projector(preds_S)

        distill_loss = []

        for i in range(4):
            loss = self.get_dis_loss(preds_S[i], preds_T[i]) * self.alpha_mfd[i]
            distill_loss.append(loss)

        return distill_loss


    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        new_fea = F.normalize(preds_S, dim=1, p=2)
        preds_T = F.normalize(preds_T, dim=1, p=2)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss




class FeatureProjector(nn.Module):

    def __init__(self,
                 student_dims,
                 teacher_dims,
                 ):    
        super(FeatureProjector, self).__init__()

        self.generator = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(student_dims[i], teacher_dims[i], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)) 
                    # nn.Conv2d(teacher_dims[i], teacher_dims[i], kernel_size=3, padding=1))

            for i in range(4)]
        )

        self.super_resoultor = nn.ModuleList(
            [
                SuperResoultor(
                    input_dims=teacher_dims[i+1],
                    output_dims=teacher_dims[i],
                    output_resoultion=56//(2**i)
                )

                for i in range(3)]
        )

        self.mixer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(teacher_dims[i]*2, teacher_dims[i], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True))

            for i in range(3)]
        )

        self.projector = nn.ModuleList(
            [
                nn.Conv2d(teacher_dims[i], teacher_dims[i], kernel_size=3, padding=1)

            for i in range(4)]
        )

        
    def forward(self, input):
        x_0, x_1, x_2, x_3 = input

        x_0 = self.generator[0](x_0)
        x_1 = self.generator[1](x_1)
        x_2 = self.generator[2](x_2)
        x_3 = self.generator[3](x_3)



        out_3 = x_3
        x_2 = self.mix_feature(x_3, x_2, 2)
        out_2 = x_2
        x_1 = self.mix_feature(x_2, x_1, 1)
        out_1 = x_1
        x_1 = self.mix_feature(x_1, x_0, 0)
        out_0 = x_0

        out_0 = self.projector[0](out_0)
        out_1 = self.projector[1](out_1)
        out_2 = self.projector[2](out_2)
        out_3 = self.projector[3](out_3)

        return [out_0, out_1, out_2, out_3]


    def mix_feature(self, feat_0, feat_1, index):
        N_0, C_0, H_0, W_0 = feat_0.shape
        N_1, C_1, H_1, W_1 = feat_1.shape

        feat_0 = self.super_resoultor[index](feat_0)
        feat_1 = self.mixer[index](torch.cat((feat_1, feat_0), dim=1))
        
        return feat_1


class SuperResoultor(nn.Module):
    def __init__(self,
                 input_dims,
                 output_dims,
                 output_resoultion,
                 ):    
        super(SuperResoultor, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.output_resoultion =output_resoultion

        self.upsampler = nn.Sequential(
            nn.Conv2d(input_dims, 4 * output_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * output_dims, 4 * output_dims, kernel_size=1, padding=0),
            )
        
        self.norm = nn.LayerNorm([output_dims, output_resoultion, output_resoultion])
    
    def forward(self, input):
        N, C_in, H_in, W_in = input.shape
        C_out, H_out, W_out = self.output_dims, H_in*2, W_in * 2

        out = self.upsampler(input)
        out = out.view(N, C_out, 2, 2, H_in, W_in).permute(0,1,4,2,5,3).contiguous().view(N, C_out, H_out, W_out)
        out = self.norm(out)

        return out




# class FeatureProjector(nn.Module):

#     def __init__(self,
#                  student_dims,
#                  teacher_dims,
#                  ):    
#         super(FeatureProjector, self).__init__()

#         self.generator = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Conv2d(student_dims[i], teacher_dims[i], kernel_size=3, padding=1),
#                     nn.ReLU(inplace=True)) 
#                     # nn.Conv2d(teacher_dims[i], teacher_dims[i], kernel_size=3, padding=1))

#             for i in range(4)]
#         )

#         self.patch_merger = nn.ModuleList(
#             [
#                 PatchMerging(
#                     in_channels=teacher_dims[i],
#                     out_channels=teacher_dims[i+1],
#                     kernel_size=(2,2),
#                     stride=2,)

#                 for i in range(3)]
#         )

#         self.mixer = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Conv2d(teacher_dims[i+1]*2, teacher_dims[i+1], kernel_size=3, padding=1),
#                     nn.ReLU(inplace=True))

#             for i in range(3)]
#         )

#         self.projector = nn.ModuleList(
#             [
#                 nn.Conv2d(teacher_dims[i], teacher_dims[i], kernel_size=3, padding=1)

#             for i in range(4)]
#         )

        
#     def forward(self, input):
#         x_0, x_1, x_2, x_3 = input

#         x_0 = self.generator[0](x_0)
#         x_1 = self.generator[1](x_1)
#         x_2 = self.generator[2](x_2)
#         x_3 = self.generator[3](x_3)

#         out_0 = x_0
#         x_1 = self.mix_feature(x_0, x_1, 0)
#         out_1 = x_1
#         x_2 = self.mix_feature(x_1, x_2, 1)
#         out_2 = x_2
#         x_3 = self.mix_feature(x_2, x_3, 2)
#         out_3 = x_3

#         out_0 = self.projector[0](out_0)
#         out_1 = self.projector[1](out_1)
#         out_2 = self.projector[2](out_2)
#         out_3 = self.projector[3](out_3)

#         return [out_0, out_1, out_2, out_3]


#     def mix_feature(self, feat_0, feat_1, index):
#         N_0, C_0, H_0, W_0 = feat_0.shape
#         N_1, C_1, H_1, W_1 = feat_1.shape

#         feat_0 = feat_0.permute(0,2,3,1).contiguous().view(N_0, -1, C_0)
#         feat_0 = self.patch_merger[index](feat_0, (H_0, W_0))[0].permute(0,2,1).view(N_1, C_1, H_1, W_1)
#         feat_1 = self.mixer[index](torch.cat((feat_1, feat_0), dim=1))
        
#         return feat_1