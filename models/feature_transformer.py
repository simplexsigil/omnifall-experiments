from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from safetensors.torch import load_file


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings in log space
        # Using 100 as default max_len to support longer sequences if needed
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register buffer (not a parameter, but part of the module)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, : x.size(1), :]


class FeatureTransformer(nn.Module):
    def __init__(
        self,
        feature_dim=2048,
        hidden_dim=768,
        num_classes=10,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        mlp_ratio=4.0,
        num_features=3,
    ):
        """
        Feature Transformer model for video feature classification

        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden dimension for transformer
            num_classes: Number of output classes
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            mlp_ratio: MLP hidden dimension ratio
            num_frames: Number of frames/features per sample
        """
        super(FeatureTransformer, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio

        # Feature projection
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim)

        # Add positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=num_features)

        # Create transformer encoder with Pre-LayerNorm for better stability
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm improves stability on small stacks
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, features):
        """
        Forward pass

        Args:
            features: Tensor of shape [batch_size, num_frames, feature_dim]

        Returns:
            logits: Tensor of shape [batch_size, num_classes]
        """
        # Project features and apply layer normalization
        x = self.feature_projection(features)
        x = self.feature_norm(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)

        # Classification
        logits = self.classifier(x)

        return {"logits": logits}

    @classmethod
    def from_pretrained(cls, model_path, cfg=None):
        """
        Load a pretrained model from a given path

        Args:
            model_path: Path to the saved model checkpoint

        Returns:
            Loaded FeatureTransformer model
        """
        # Load the checkpoint
        # Try loading with safetensors first
        if model_path.endswith(".safetensors"):
            checkpoint = load_file(model_path)
        else:
            # Fall back to regular PyTorch loading
            checkpoint = torch.load(model_path, map_location="cpu")

        # Extract model configuration if it exists, otherwise use default values
        if "config" in checkpoint or cfg is not None:
            cfg = checkpoint["config"] if "config" in checkpoint else cfg

            model = cls(
                feature_dim=cfg.get("feature_dim", 2048),
                hidden_dim=cfg.get("hidden_dim", 768),
                num_classes=cfg.get("num_classes", 10),
                num_layers=cfg.get("num_layers", 2),
                num_heads=cfg.get("num_heads", 8),
                dropout=cfg.get("dropout", 0.1),
                mlp_ratio=cfg.get("mlp_ratio", 4.0),
                num_features=cfg.get("num_frames", 3),
            )
        else:
            # If no config, check if the state dict has metadata to infer parameters
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            default_feature_dim = 2048
            default_hidden_dim = 768
            default_num_classes = 10
            default_num_layers = 2
            default_num_heads = 8
            default_dropout = 0.1
            default_mlp_ratio = 4.0
            default_num_features = 3

            feature_dim = (
                state_dict["feature_projection.weight"].size(1)
                if "feature_projection.weight" in state_dict
                else default_feature_dim
            )
            hidden_dim = (
                state_dict["feature_projection.weight"].size(0)
                if "feature_projection.weight" in state_dict
                else default_hidden_dim
            )
            num_classes = (
                state_dict["classifier.weight"].size(0) if "classifier.weight" in state_dict else default_num_classes
            )

            # Initialize potentially inferable parameters with defaults
            num_layers = default_num_layers
            mlp_ratio = default_mlp_ratio
            num_features = default_num_features

            # Try to infer num_layers
            # Counts the number of 'transformer_encoder.layers.X' present.
            layer_indices = set()
            for k in state_dict.keys():
                if k.startswith("transformer_encoder.layers."):
                    parts = k.split(".")
                    # Expecting keys like transformer_encoder.layers.0.self_attn...
                    if len(parts) > 2 and parts[2].isdigit():
                        layer_indices.add(int(parts[2]))
            if layer_indices:
                num_layers = max(layer_indices) + 1

            # Try to infer mlp_ratio
            # Requires hidden_dim to be known (either inferred or default)
            # And transformer_encoder.layers.0.linear1.weight to exist
            if "transformer_encoder.layers.0.linear1.weight" in state_dict:
                dim_feedforward = state_dict["transformer_encoder.layers.0.linear1.weight"].size(0)
                if hidden_dim > 0:  # Ensure hidden_dim is valid before division
                    mlp_ratio = dim_feedforward / hidden_dim

            # Try to infer num_features (from pos_encoder.pe's sequence length)
            if "pos_encoder.pe" in state_dict:
                # pos_encoder.pe has shape [1, max_len, d_model]
                num_features = state_dict["pos_encoder.pe"].size(1)

            # Parameters not inferred from state_dict (use defaults)
            # num_heads is difficult to infer reliably from state_dict shapes alone.
            num_heads = default_num_heads
            # dropout is a rate and not stored in state_dict weights.
            dropout = default_dropout

            # Create model with inferred parameters and default values for others
            model = cls(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                num_features=num_features,  # Matches __init__ parameter name
            )

        # Load the state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        return model


class FeatureTransformerWrapper(nn.Module):
    """
    Wrapper around FeatureTransformer to make it compatible with HuggingFace Trainer.
    """

    def __init__(self, model: FeatureTransformer):
        super().__init__()
        self.model = model

    def forward(
        self, features: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass through the model with optional loss computation.

        Args:
            features: Input features tensor of shape [batch_size, num_frames, feature_dim]
            labels: Class labels tensor of shape [batch_size]
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing model outputs including logits and optional loss
        """
        outputs = self.model(features)

        return outputs

    @classmethod
    def from_pretrained(cls, model_path: str) -> "FeatureTransformerWrapper":
        """
        Load a pretrained model from a given path.

        Args:
            model_path: Path to the saved model checkpoint

        Returns:
            Loaded FeatureTransformerWrapper model
        """
        # Create a basic FeatureTransformer model (will be overridden with loaded parameters)
        feature_transformer = FeatureTransformer()

        # Load the state dict
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path)
        else:
            # Fall back to regular PyTorch loading
            checkpoint = torch.load(model_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

        # Load the state dict with non-strict loading
        result = feature_transformer.load_state_dict(state_dict, strict=False)

        # Log warnings for missing or unexpected keys
        if result.missing_keys:
            print(f"Warning: Missing keys when loading checkpoint: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Warning: Unexpected keys when loading checkpoint: {result.unexpected_keys}")

        # Create and return the wrapper
        return cls(model=feature_transformer)
