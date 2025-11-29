"""
Model modules for H-MTL framework

This package contains all neural network components:
- CNNBackbone: 1D CNN feature extractor
- SeverityPatternFusion: SPF module for severity encoding
- IterativeKnowledgeRefinement: IKR module for knowledge transfer
- H_MTL_Model: Main hierarchical multi-task learning model
"""

from .backbone import CNNBackbone
from .spf_module import (
    SeverityEmbedding,
    DomainSpecificFeatureExtractor,
    SeverityPatternFusion
)
from .ikr_module import (
    SingleKnowledgeTransferStep,
    IterativeKnowledgeRefinement
)
from .h_mtl_model import H_MTL_Model

__all__ = [
    # Backbone
    'CNNBackbone',
    
    # SPF Module components
    'SeverityEmbedding',
    'DomainSpecificFeatureExtractor',
    'SeverityPatternFusion',
    
    # IKR Module components
    'SingleKnowledgeTransferStep',
    'IterativeKnowledgeRefinement',
    
    # Main model
    'H_MTL_Model',
]

__version__ = "1.0.0"
__description__ = "H-MTL model components"
