"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    n_members: int = 3_000
    service_year: int = 2025            # diagnoses come from this year (prospective model)
    payment_year: int = 2026            # the score is used to pay this year
    blend_year: int = 2025              # demo also shows a blended year (33% V24 / 67% V28)
    normalization_factor: float = 1.05  # ILLUSTRATIVE — real value is in the CMS Rate Announcement
    coding_intensity: float = 0.059     # statutory minimum MA coding-pattern adjustment
    ridge: float = 1.0                  # ALTERNATIVE: penalty for empirical weights
    train_frac: float = 0.5             # ALTERNATIVE: fit on this share, validate on the rest
    documentation_rate: float = 0.8     # synthetic: share of true conditions coded acceptably
