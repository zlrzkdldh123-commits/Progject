"""
Iterative Knowledge Refinement (IKR) Module
Bidirectional knowledge transfer between main and sub tasks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleKnowledgeTransferStep(nn.Module):
    """Single iteration of knowledge transfer (Step 1-3)"""
    def __init__(self, hidden_dim, num_heads=2):
        super().__init__()
        
        # Step 1: Feature Exchange (Linear transformation)
        self.main_to_tension = nn.Linear(hidden_dim, hidden_dim)
        self.main_to_wear = nn.Linear(hidden_dim, hidden_dim)
        self.tension_to_main = nn.Linear(hidden_dim, hidden_dim)
        self.wear_to_main = nn.Linear(hidden_dim, hidden_dim)
        
        # Step 2: Multi-head attention for nonlinear dependencies
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Step 3: Residual connections
        self.residual_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, main_feat, tension_feat, wear_feat):
        """
        Args:
            main_feat: (batch_size, hidden_dim)
            tension_feat: (batch_size, hidden_dim)
            wear_feat: (batch_size, hidden_dim)
        
        Returns:
            enhanced_main: (batch_size, hidden_dim)
            enhanced_tension: (batch_size, hidden_dim)
            enhanced_wear: (batch_size, hidden_dim)
        """
        batch_size = main_feat.size(0)
        
        # Step 1: Feature Exchange
        main_to_t = self.main_to_tension(main_feat)
        main_to_w = self.main_to_wear(main_feat)
        t_to_main = self.tension_to_main(tension_feat)
        w_to_main = self.wear_to_main(wear_feat)
        
        # Step 2: Multi-head Attention
        # Stack task features for attention
        task_features = torch.stack([main_feat, tension_feat, wear_feat], dim=1)  
        # (batch_size, 3, hidden_dim)
        
        attended, _ = self.attention(task_features, task_features, task_features)
        # (batch_size, 3, hidden_dim)
        
        attended_main = attended[:, 0, :]
        attended_tension = attended[:, 1, :]
        attended_wear = attended[:, 2, :]
        
        # Step 3: Residual Update
        enhanced_main = main_feat + t_to_main + w_to_main + attended_main
        enhanced_tension = tension_feat + main_to_t + attended_tension
        enhanced_wear = wear_feat + main_to_w + attended_wear
        
        # Layer normalization for stability
        enhanced_main = self.residual_norm(enhanced_main)
        enhanced_tension = self.residual_norm(enhanced_tension)
        enhanced_wear = self.residual_norm(enhanced_wear)
        
        return enhanced_main, enhanced_tension, enhanced_wear


class IterativeKnowledgeRefinement(nn.Module):
    """
    IKR Module: K iterations of bidirectional knowledge refinement
    
    Args:
        hidden_dim: feature dimension
        num_iterations: number of refinement iterations (default: 3)
    """
    def __init__(self, hidden_dim=128, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations
        self.hidden_dim = hidden_dim
        
        # K knowledge transfer layers
        self.transfer_layers = nn.ModuleList([
            SingleKnowledgeTransferStep(hidden_dim)
            for _ in range(num_iterations)
        ])
        
        # Learnable weights for combining outputs from K iterations
        self.iteration_weights = nn.Parameter(torch.ones(num_iterations))
    
    def forward(self, main_feat, tension_feat, wear_feat):
        """
        Args:
            main_feat: (batch_size, hidden_dim) - fault type classification features
            tension_feat: (batch_size, hidden_dim) - tension severity features
            wear_feat: (batch_size, hidden_dim) - wear severity features
        
        Returns:
            refined_dict: dictionary with refined features and refinement info
        """
        current_main = main_feat
        current_tension = tension_feat
        current_wear = wear_feat
        
        iteration_results = []
        
        # Execute K iterations
        for k, transfer_layer in enumerate(self.transfer_layers):
            enhanced_main, enhanced_tension, enhanced_wear = transfer_layer(
                current_main, current_tension, current_wear
            )
            
            iteration_results.append({
                'iteration': k + 1,
                'main': enhanced_main,
                'tension': enhanced_tension,
                'wear': enhanced_wear
            })
            
            # Update for next iteration
            current_main = enhanced_main
            current_tension = enhanced_tension
            current_wear = enhanced_wear
        
        # Aggregate outputs from all iterations with learned weights
        iteration_weights = F.softmax(self.iteration_weights, dim=0)
        
        final_main = torch.zeros_like(main_feat)
        final_tension = torch.zeros_like(tension_feat)
        final_wear = torch.zeros_like(wear_feat)
        
        for w, result in zip(iteration_weights, iteration_results):
            final_main = final_main + w * result['main']
            final_tension = final_tension + w * result['tension']
            final_wear = final_wear + w * result['wear']
        
        return {
            'final_main': final_main,
            'final_tension': final_tension,
            'final_wear': final_wear,
            'iteration_results': iteration_results,
            'iteration_weights': iteration_weights,
            'num_iterations': self.num_iterations
        }
