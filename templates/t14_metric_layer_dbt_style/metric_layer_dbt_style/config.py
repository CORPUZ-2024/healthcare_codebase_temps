"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    seed: int = 14
    n_members: int = 20_000
    engagement_control: float = 0.22       # outreach A/B test: new script (treatment) vs current
    engagement_lift: float = 0.03
    pmpm_effect_pct: float = -0.03         # small true cost effect: CUPED helps detect it
    complaint_control: float = 0.010
    complaint_lift: float = 0.002          # guardrail: slightly worse, inside the 0.5 pp margin
    models_dir: Path = HERE / "models"
    schema_yml: Path = HERE / "models" / "schema.yml"
    metrics_yml: Path = HERE / "metrics.yaml"
    alpha: float = 0.05
    srm_alpha: float = 0.001
