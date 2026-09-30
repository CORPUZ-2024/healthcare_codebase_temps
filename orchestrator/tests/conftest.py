"""Make the orchestrator modules importable in its own tests (never template code)."""
import sys
from pathlib import Path

ORCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORCH))
REPO = ORCH.parent
