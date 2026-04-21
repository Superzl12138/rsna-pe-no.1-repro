import os

import torch
import torch.nn as nn


def get_relation_head_config():
    enabled = os.environ.get("ENABLE_RELATION_HEAD", "0") == "1"
    hidden_dim = int(os.environ.get("RELATION_HEAD_HIDDEN_DIM", "64"))
    if hidden_dim <= 0:
        raise ValueError(f"RELATION_HEAD_HIDDEN_DIM must be positive, got {hidden_dim}")
    return {
        "enabled": enabled,
        "hidden_dim": hidden_dim,
    }


def print_relation_head_config(config):
    print("=" * 60)
    print("2nd-level Relation Head")
    print("=" * 60)
    print(f"enabled: {config['enabled']}")
    print(f"hidden_dim: {config['hidden_dim']}")
    print("=" * 60)


class RelationAwareSubtypeHead(nn.Module):
    def __init__(self, feature_dim, context_dim, hidden_dim):
        super().__init__()
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.chronic_head = nn.Linear(feature_dim + hidden_dim, 1)
        self.acute_and_chronic_head = nn.Linear(feature_dim + hidden_dim, 1)

    def forward(self, conc, context_logits):
        context_feature = self.context_mlp(context_logits)
        fused_feature = torch.cat([conc, context_feature], dim=1)
        logits_chronic = self.chronic_head(fused_feature)
        logits_acute_and_chronic = self.acute_and_chronic_head(fused_feature)
        return logits_chronic, logits_acute_and_chronic
