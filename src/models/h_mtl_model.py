"""
Hierarchical Severity-Aware Multi-Task Learning (H-MTL) Model
For semiconductor transfer robot belt fault diagnosis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import CNNBackbone
from .spf_module import SeverityPatternFusion
from .ikr_module import IterativeKnowledgeRefinement


class H_MTL_Model(nn.Module):
    """
    Main H-MTL Architecture combining:
    - CNN Backbone for feature extraction
    - SPF Module for severity-aware representations
    - IKR Module for iterative knowledge refinement
    
    Main Task: Fault-type classification (Normal, Tension, Wear)
    Sub Tasks: Severity estimation (Light, Medium, Severe) for each fault type
    """
    
    def __init__(self, seq_len=780, hidden_dim=128, num_iterations=3):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        
        # 1. CNN Backbone
        self.backbone = CNNBackbone(input_channels=2, hidden_dim=hidden_dim)
        
        # 2. SPF Module - Severity Pattern Fusion
        self.spf = SeverityPatternFusion(input_dim=hidden_dim, hidden_dim=hidden_dim)
        
        # 3. IKR Module - Iterative Knowledge Refinement
        self.ikr = IterativeKnowledgeRefinement(hidden_dim=hidden_dim, 
                                               num_iterations=num_iterations)
        
        # Task-specific processors
        self.main_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.tension_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.wear_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Task-specific classifiers
        # Main task: 3 classes (Normal, Tension, Wear)
        self.main_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # Sub tasks: 3 severity levels each (Light, Medium, Severe)
        self.tension_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self.wear_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x, stage='both'):
        """
        Args:
            x: (batch_size, 2, 780) - vibration signal
            stage: 'both' (main+sub), 'main' (only main), or 'sub' (only sub)
        
        Returns:
            main_logits: (batch_size, 3) - fault type predictions
            tension_logits: (batch_size, 3) or None - tension severity
            wear_logits: (batch_size, 3) or None - wear severity
            auxiliary_outputs: dict with intermediate representations
        """
        # 1. CNN Backbone
        h = self.backbone(x)  # (batch_size, hidden_dim)
        
        # 2. SPF Module
        spf_outputs = self.spf(h)
        integrated_knowledge = spf_outputs['integrated_knowledge']
        tension_base = spf_outputs['tension_features']
        wear_base = spf_outputs['wear_features']
        
        # 3. Task-specific processors
        main_feat = self.main_processor(torch.cat([h, integrated_knowledge], dim=-1))
        tension_feat = self.tension_processor(tension_base)
        wear_feat = self.wear_processor(wear_base)
        
        # 4. IKR Module - Iterative Refinement
        ikr_outputs = self.ikr(main_feat, tension_feat, wear_feat)
        
        refined_main = ikr_outputs['final_main']
        refined_tension = ikr_outputs['final_tension']
        refined_wear = ikr_outputs['final_wear']
        
        # 5. Task-specific classifiers
        main_logits = self.main_classifier(refined_main)
        
        tension_logits = None
        wear_logits = None
        
        if stage in ['both', 'sub']:
            tension_logits = self.tension_classifier(refined_tension)
            wear_logits = self.wear_classifier(refined_wear)
        
        # Auxiliary outputs for loss computation and analysis
        auxiliary_outputs = {
            'spf_outputs': spf_outputs,
            'ikr_outputs': ikr_outputs,
            'main_feat': main_feat,
            'tension_feat': tension_feat,
            'wear_feat': wear_feat,
            'refined_main': refined_main,
            'refined_tension': refined_tension,
            'refined_wear': refined_wear
        }
        
        return main_logits, tension_logits, wear_logits, auxiliary_outputs
    
    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_dim': self.hidden_dim,
            'num_iterations': self.num_iterations,
            'backbone': 'CNN (3 blocks)',
            'main_task_classes': 3,
            'sub_task_classes': 3,
            'modules': ['Backbone', 'SPF', 'IKR']
        }
