import os

import torch
import torch.nn as nn


def get_relation_head_config():
    enabled = os.environ.get("ENABLE_RELATION_HEAD", "0") == "1"
    relation_type = os.environ.get("RELATION_HEAD_TYPE", "mlp")
    if relation_type not in {"mlp", "graph"}:
        raise ValueError(f"Unsupported RELATION_HEAD_TYPE: {relation_type}")
    hidden_dim = int(os.environ.get("RELATION_HEAD_HIDDEN_DIM", "64"))
    if hidden_dim <= 0:
        raise ValueError(f"RELATION_HEAD_HIDDEN_DIM must be positive, got {hidden_dim}")
    num_steps = int(os.environ.get("RELATION_GRAPH_STEPS", "2"))
    if num_steps <= 0:
        raise ValueError(f"RELATION_GRAPH_STEPS must be positive, got {num_steps}")
    return {
        "enabled": enabled,
        "relation_type": relation_type,
        "hidden_dim": hidden_dim,
        "num_steps": num_steps,
    }


def print_relation_head_config(config):
    print("=" * 60)
    print("2nd-level Relation Head")
    print("=" * 60)
    print(f"enabled: {config['enabled']}")
    print(f"type: {config['relation_type']}")
    print(f"hidden_dim: {config['hidden_dim']}")
    print(f"graph_steps: {config['num_steps']}")
    print("=" * 60)


class MLPRelationAwareSubtypeHead(nn.Module):
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


class LabelGraphSubtypeHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_steps):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.num_nodes = 9
        self.node_encoder = nn.Linear(1, hidden_dim)
        self.feature_to_nodes = nn.Linear(feature_dim, self.num_nodes * hidden_dim)
        self.message_linear = nn.Linear(hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, 1)

        adjacency = torch.eye(self.num_nodes, dtype=torch.float32)
        # Soft priors: global negative with all PE-related labels
        adjacency[0, 1:] = 1.0
        adjacency[1:, 0] = 1.0
        # Spatial/location labels and RV/LV weakly support subtype reasoning
        for idx in [2, 3, 4, 5, 6]:
            adjacency[idx, 7] = 1.0
            adjacency[idx, 8] = 1.0
            adjacency[7, idx] = 1.0
            adjacency[8, idx] = 1.0
        # Strong prior between chronic and acute_and_chronic
        adjacency[7, 8] = 1.0
        adjacency[8, 7] = 1.0
        self.register_buffer("adjacency", adjacency)

    def forward(self, conc, all_exam_logits):
        batch_size = conc.size(0)
        node_states = self.node_encoder(all_exam_logits.unsqueeze(-1))
        feature_bias = self.feature_to_nodes(conc).view(batch_size, self.num_nodes, self.hidden_dim)
        node_states = node_states + feature_bias

        adjacency = self.adjacency / self.adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        for _ in range(self.num_steps):
            message = torch.matmul(adjacency.unsqueeze(0), self.message_linear(node_states))
            node_states = torch.tanh(self.update_linear(torch.cat([node_states, message], dim=-1)))

        logits_chronic = self.output_linear(node_states[:, 7, :])
        logits_acute_and_chronic = self.output_linear(node_states[:, 8, :])
        return logits_chronic, logits_acute_and_chronic


def build_relation_subtype_head(feature_dim, context_dim, config):
    relation_type = config["relation_type"]
    if relation_type == "mlp":
        return MLPRelationAwareSubtypeHead(
            feature_dim=feature_dim,
            context_dim=context_dim,
            hidden_dim=config["hidden_dim"],
        )
    return LabelGraphSubtypeHead(
        feature_dim=feature_dim,
        hidden_dim=config["hidden_dim"],
        num_steps=config["num_steps"],
    )
