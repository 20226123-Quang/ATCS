"""Load and validate model configuration for ACAC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelSettings:
    tls_names: list[str]
    num_agents: int
    macro_obs_dim: int
    macro_action_dim: int
    hidden_dim: int
    time_embed_dim: int
    attn_dim: int
    num_heads: int


@dataclass(frozen=True)
class TrainingSettings:
    buffer_size: int
    actor_lr: float
    critic_lr: float
    gamma: float
    lam: float
    eps_clip: float
    eps: float
    vf_coef: float  # c1: value function loss coefficient
    ent_coef: float  # c2: entropy bonus coefficient
    reward_delay_weight: float
    reward_queue_weight: float
    reward_saturation_weight: float


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    model: ModelSettings
    training: TrainingSettings


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "model_config.json"


def load_model_config(config_path: Optional[str] = None) -> ModelConfig:
    """Load model config JSON and convert to typed dataclasses."""
    path = Path(config_path) if config_path else _default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    m = raw.get("model", {})
    model = ModelSettings(
        tls_names=list(m.get("tls_names", ["E_1", "E_2", "E_3", "E_4"])),
        num_agents=int(m.get("num_agents", 4)),
        macro_obs_dim=int(m.get("macro_obs_dim", 32)),
        macro_action_dim=int(m.get("macro_action_dim", 6)),
        hidden_dim=int(m.get("hidden_dim", 128)),
        time_embed_dim=int(m.get("time_embed_dim", 32)),
        attn_dim=int(m.get("attn_dim", 128)),
        num_heads=int(m.get("num_heads", 4)),
    )

    t = raw.get("training", {})
    training = TrainingSettings(
        buffer_size=int(t.get("buffer_size", 10000)),
        actor_lr=float(t.get("actor_lr", 1e-4)),
        critic_lr=float(t.get("critic_lr", 1e-3)),
        gamma=float(t.get("gamma", 0.99)),
        lam=float(t.get("lam", 0.95)),
        eps_clip=float(t.get("eps_clip", 0.2)),
        eps=float(t.get("eps", 1e-6)),
        vf_coef=float(t.get("vf_coef", 0.5)),
        ent_coef=float(t.get("ent_coef", 0.01)),
        reward_delay_weight=float(t.get("reward_delay_weight", 1.0)),
        reward_queue_weight=float(t.get("reward_queue_weight", 0.2)),
        reward_saturation_weight=float(t.get("reward_saturation_weight", 0.1)),
    )

    return ModelConfig(path=path, model=model, training=training)
