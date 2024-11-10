from .kd import KDLoss
from .fd import FDLoss
from .fmd import FreqMaskingDistillLoss
from .fmdv2 import FreqMaskingDistillLossv2
from .fmd_convnext import FreqMaskingDistillLossConvNext


__all__ = [
     'KDLoss', 'FDLoss' , 'FreqMaskingDistillLoss', 'FreqMaskingDistillLossv2', 'FreqMaskingDistillLossConvNext'
]
