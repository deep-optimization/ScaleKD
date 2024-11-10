# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .hugging_face import HuggingFaceClassifier
from .image import ImageClassifier
from .timm import TimmClassifier
from .distiller import ClassificationDistiller
from .multi_layer_distiller import MultiLayerClassificationDistiller

__all__ = [
    'BaseClassifier', 'ImageClassifier', 'TimmClassifier',
    'HuggingFaceClassifier', 'ClassificationDistiller','MultiLayerClassificationDistiller'
]
