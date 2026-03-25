from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    return nn.ReLU()


class BaseAnomalyModel(nn.Module):
    model_name = "base"

    def compute_loss(self, outputs: dict[str, torch.Tensor], inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def reconstruction_error(self, outputs: dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
        return torch.mean((outputs["reconstruction"] - inputs) ** 2, dim=1)


class EnhancedMLPAutoencoder(BaseAnomalyModel):
    model_name = "enhanced_mlp_ae"

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation: str = "relu",
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        act = _activation(activation)

        encoder_layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(copy.deepcopy(act))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        encoder_layers.append(nn.Linear(previous_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        previous_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(copy.deepcopy(act))
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(inputs)
        reconstruction = self.decoder(latent)
        return {"reconstruction": reconstruction, "latent": latent}

    def compute_loss(self, outputs: dict[str, torch.Tensor], inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        reconstruction_loss = F.mse_loss(outputs["reconstruction"], inputs)
        return {"loss": reconstruction_loss, "reconstruction_loss": reconstruction_loss}


class HybridScoringAutoencoder(EnhancedMLPAutoencoder):
    model_name = "hybrid_ae"


class TransformerAutoencoder(BaseAnomalyModel):
    model_name = "transformer_ae"

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_embed = nn.Linear(1, d_model)
        self.feature_positional = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim), nn.GELU())
        self.latent_to_token = nn.Linear(latent_dim, d_model)
        self.decoder_block = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.feature_embed(inputs.unsqueeze(-1)) + self.feature_positional.unsqueeze(0)
        encoded = self.encoder(tokens)
        latent = self.to_latent(encoded.mean(dim=1))
        latent_context = self.latent_to_token(latent).unsqueeze(1)
        decoded_tokens = self.decoder_block(encoded + latent_context)
        reconstruction = self.output_head(decoded_tokens).squeeze(-1)
        return {"reconstruction": reconstruction, "latent": latent}

    def compute_loss(self, outputs: dict[str, torch.Tensor], inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        reconstruction_loss = F.mse_loss(outputs["reconstruction"], inputs)
        return {"loss": reconstruction_loss, "reconstruction_loss": reconstruction_loss}


class VariationalAutoencoder(BaseAnomalyModel):
    model_name = "vae"

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation: str = "relu",
        dropout: float = 0.1,
        beta: float = 0.05,
    ) -> None:
        super().__init__()
        act = _activation(activation)
        encoder_layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            encoder_layers.append(copy.deepcopy(act))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu_head = nn.Linear(previous_dim, latent_dim)
        self.logvar_head = nn.Linear(previous_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        previous_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            decoder_layers.append(copy.deepcopy(act))
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.beta = beta

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(inputs)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        latent = self._reparameterize(mu, logvar)
        reconstruction = self.decoder(latent)
        return {"reconstruction": reconstruction, "latent": mu, "mu": mu, "logvar": logvar}

    def compute_loss(self, outputs: dict[str, torch.Tensor], inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        reconstruction_loss = F.mse_loss(outputs["reconstruction"], inputs)
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = reconstruction_loss + self.beta * kl_loss
        return {"loss": total, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}


MODEL_REGISTRY = {
    EnhancedMLPAutoencoder.model_name: EnhancedMLPAutoencoder,
    TransformerAutoencoder.model_name: TransformerAutoencoder,
    VariationalAutoencoder.model_name: VariationalAutoencoder,
    HybridScoringAutoencoder.model_name: HybridScoringAutoencoder,
}


def resolve_model_config(config: dict[str, Any], model_name: str | None = None) -> tuple[str, dict[str, Any]]:
    selected_name = model_name or config["model"]["name"]
    if selected_name not in config["models"]:
        raise ValueError(f"Unknown model config: {selected_name}")
    resolved = copy.deepcopy(config["models"][selected_name])
    resolved["name"] = selected_name
    return selected_name, resolved


def create_model(model_name: str, input_dim: int, config: dict[str, Any]) -> tuple[BaseAnomalyModel, dict[str, Any]]:
    selected_name, model_cfg = resolve_model_config(config, model_name)
    model_type = model_cfg.get("type", selected_name)
    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unsupported model type: {model_type}")

    kwargs = {key: value for key, value in model_cfg.items() if key not in {"type", "name", "scoring"}}
    model = model_cls(input_dim=input_dim, **kwargs)
    return model, model_cfg


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
