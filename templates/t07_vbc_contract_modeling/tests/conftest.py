import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vbc_contract_modeling import data  # noqa: E402
from vbc_contract_modeling.config import Config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def contracts(cfg):
    return data.load_contracts(cfg.contracts_dir)


@pytest.fixture(scope="session")
def enh(contracts):
    return contracts["FAKE_MSSP_ENHANCED"]


@pytest.fixture(scope="session")
def basic(contracts):
    return contracts["FAKE_MSSP_BASIC_A"]
