"""Load and validate model and scenario configuration for ACAC."""

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
    vf_coef: float
    ent_coef: float
    reward_delay_weight: float
    reward_queue_weight: float
    reward_saturation_weight: float
    reward_split_failure_weight: float
    reward_starvation_weight: float
    reward_fairness_weight: float


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    model: ModelSettings
    training: TrainingSettings


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    sumocfg_path: Path
    checkpoint_path: Path
    tags: tuple[str, ...]
    enabled: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "model_config.json"


def _default_scenario_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "scenario_config.json"


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (_repo_root() / path).resolve()


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
        reward_split_failure_weight=float(t.get("reward_split_failure_weight", 0.75)),
        reward_starvation_weight=float(t.get("reward_starvation_weight", 0.5)),
        reward_fairness_weight=float(t.get("reward_fairness_weight", 0.5)),
    )

    return ModelConfig(path=path, model=model, training=training)


def load_scenario_config(
    config_path: Optional[str] = None,
    only_enabled: bool = True,
) -> list[ScenarioConfig]:
    """Load scenario registry and resolve paths relative to the repository root."""
    path = Path(config_path) if config_path else _default_scenario_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Scenario config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    scenarios: list[ScenarioConfig] = []
    for item in raw.get("scenarios", []):
        enabled = bool(item.get("enabled", True))
        if only_enabled and not enabled:
            continue

        name = str(item["name"]).strip()
        sumocfg_path = _resolve_repo_path(str(item["sumocfg"]))
        checkpoint_path = _resolve_repo_path(str(item["checkpoint"]))
        tags = tuple(str(tag).strip() for tag in item.get("tags", []))

        scenarios.append(
            ScenarioConfig(
                name=name,
                sumocfg_path=sumocfg_path,
                checkpoint_path=checkpoint_path,
                tags=tags,
                enabled=enabled,
            )
        )

    return scenarios
