from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcls.models import build_classifier
from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from mmengine.model import revert_sync_batchnorm

from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from mmengine.runner.checkpoint import  load_checkpoint, _load_checkpoint, load_state_dict

@MODELS.register_module()
class ClassificationDistiller(BaseModel, metaclass=ABCMeta):
    """Base distiller for dis_classifiers.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 is_vit = False,
                 use_logit = False,
                 sd = False,
                 distill_cfg = None,
                 teacher_pretrained = None,
                 sync_bn=False,
                 with_cls_token=False,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')

        if train_cfg is not None and 'augments' in train_cfg:
            # Set batch augmentations by `train_cfg`
            data_preprocessor['batch_augments'] = train_cfg

        super(ClassificationDistiller, self).__init__(
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.teacher = build_classifier((Config.fromfile(teacher_cfg)).model)
        self.teacher_pretrained = teacher_pretrained
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            

        self.student = build_classifier((Config.fromfile(student_cfg)).model)
        if sync_bn:
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            print(self.student)

        self.distill_cfg = distill_cfg   
        self.distill_losses = nn.ModuleDict()
        if self.distill_cfg is not None:  
            for item_loc in distill_cfg:
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    use_this = item_loss.use_this
                    if use_this:
                        self.distill_losses[loss_name] = MODELS.build(item_loss)

        self.is_vit = is_vit
        self.sd = sd
        self.use_logit = use_logit
        self.with_cls_token = with_cls_token

        if 'loss_tcd' in self.distill_losses.keys():
            self.distill_losses['loss_tcd'].set_head(self.teacher.head)
        
        self.linear = nn.Linear(384, 384) 

    def init_weights(self):
        if self.teacher_pretrained is not None:
            # load_checkpoint(self, self.teacher_pretrained, map_location='cpu')
            load_checkpoint(self.teacher, self.teacher_pretrained, map_location='cpu')
        self.student.init_weights()

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            feats = self.student.extract_feat(inputs)
            return self.student.head(feats) if self.student.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.student.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def loss(self, inputs: torch.Tensor,
             data_samples: List[ClsDataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if 'score' in data_samples[0].gt_label:
            # Batch augmentation may convert labels to one-hot format scores.
            gt_label = torch.stack([i.gt_label.score for i in data_samples])
        else:
            gt_label = torch.cat([i.gt_label.label for i in data_samples])

        fea_s = self.student.extract_feat(inputs, stage='backbone')    

        x = fea_s
        if self.student.with_neck:
            x = self.student.neck(x)
        if self.student.with_head and hasattr(self.student.head, 'pre_logits'):
            x = self.student.head.pre_logits(x)


        logit_s = self.student.head.fc(x)
        loss = self.student.head._get_loss(logit_s, data_samples)

        s_loss = dict()
        for key in loss.keys():
            s_loss['ori_'+key] = loss[key]
        

        inputs_t = inputs

        with torch.no_grad():
            fea_t = self.teacher.extract_feat(inputs_t, stage='backbone')
            if self.use_logit:
                if self.is_vit:
                    logit_t = self.teacher.head.fc(fea_t[-1])
                    # logit_t = self.teacher.head.layers.head(fea_t[-1])
                else:
                    logit_t = self.teacher.head.fc(self.teacher.neck(fea_t[-1]))
        
        all_keys = self.distill_losses.keys()

        # feat_S_s1, feat_S_s2, feat_S_s3, feat_S_s4 = fea_s[0], fea_s[1], fea_s[2], fea_s[3]

        
        # feat_S_s1, feat_S_s2, feat_S_s3, feat_S_s4 = fea_s[0], fea_s[1], fea_s[-2], fea_s[-1]
        
        feat_T_s3, feat_T_s4 = fea_t[-2], fea_t[-1]
        feat_S_s1, feat_S_s2, feat_S_s3, feat_S_s4 = fea_s[0], fea_s[1], fea_s[2], fea_s[3]
        # feat_T_s1, feat_T_s2, feat_T_s3, feat_T_s4 = fea_t[0], fea_t[1], fea_t[2], fea_t[3]


        if 'loss_fd' in all_keys:
            loss_name = 'loss_fd'
            preds_S = fea_s[-1]
            preds_T = fea_t[-1]
            s_loss[loss_name] = self.distill_losses[loss_name](preds_S, preds_T)


        feat_S_s3_spat_query, feat_S_s3_freq_query = None, None

        if 'loss_fmd_s3' in all_keys:
            loss_name = 'loss_fmd_s3'

            if self.distill_losses[loss_name].self_query:
                query = None
            else:
                query = feat_T_s3
            
                
            feat_S_s3_spat = self.distill_losses[loss_name].project_feat_spat(feat_S_s3, query=None)
            feat_S_s3_freq = self.distill_losses[loss_name].project_feat_freq(feat_S_s3, query=None)

            feat_S_s3_spat = self.teacher.backbone.forward_specific_stage(feat_S_s3_spat, 4)
            feat_S_s3_freq = self.teacher.backbone.forward_specific_stage(feat_S_s3_freq, 4)
            
            feat_S_s3_spat_query, feat_S_s3_freq_query = feat_S_s3_spat, feat_S_s3_freq


            s_loss['loss_s3'] = self.distill_losses[loss_name].get_spat_loss(feat_S_s3_spat, feat_T_s4)
            s_loss['loss_s3_alt'] = self.distill_losses[loss_name].get_freq_loss(feat_S_s3_freq, feat_T_s4)


        if 'loss_fmd_s4' in all_keys:
            loss_name = 'loss_fmd_s4'
            if self.is_vit:
                feat_S_s4_spat = self.distill_losses[loss_name].project_feat_spat(feat_S_s4, query=feat_S_s3_spat_query)
                feat_S_s4_freq = self.distill_losses[loss_name].project_feat_freq(feat_S_s4, query=feat_S_s3_freq_query)

                feat_S_s4_spat = self.teacher.backbone.forward_specific_stage(feat_S_s4_spat, 5)
                feat_S_s4_freq = self.teacher.backbone.forward_specific_stage(feat_S_s4_freq, 5)

                s_loss['loss_s4'] = self.distill_losses[loss_name].get_spat_loss(feat_S_s4_spat, feat_T_s4)
                s_loss['loss_s4_alt'] = self.distill_losses[loss_name].get_freq_loss(feat_S_s4_freq, feat_T_s4)
            else:
                s_loss['loss_s4'], s_loss['loss_s4_alt']= self.distill_losses[loss_name](feat_S_s4, feat_T_s4, query_s=feat_S_s3_spat_query, query_f=feat_S_s3_freq_query)


        if ('loss_kd' in all_keys) and self.use_logit:
            loss_name = 'loss_kd'
            ori_alpha, s_loss[loss_name] = self.distill_losses[loss_name](logit_s, logit_t)
            s_loss['ori_loss'] = ori_alpha * s_loss['ori_loss']


        return s_loss