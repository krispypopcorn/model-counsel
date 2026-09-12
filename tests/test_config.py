from __future__ import annotations

import pytest

from model_counsel.config import CHECKPOINTING_ENABLED, env_bool, env_float, env_int


def test_boolean_configuration_is_strict() -> None:
    assert env_bool("FLAG", environ={"FLAG": "yes"}) is True
    assert env_bool("FLAG", environ={"FLAG": "OFF"}) is False
    assert env_bool("FLAG", default=True, environ={}) is True
    with pytest.raises(ValueError, match="FLAG must be one of"):
        env_bool("FLAG", environ={"FLAG": "sometimes"})


def test_numeric_configuration_is_parsed_and_clamped() -> None:
    assert env_int("LIMIT", 5, 1, 10, {"LIMIT": "99"}) == 10
    assert env_int("LIMIT", 5, 1, 10, {"LIMIT": "-4"}) == 1
    assert env_float("BUDGET", 5.0, 0.5, 10.0, {"BUDGET": "2.75"}) == 2.75
    with pytest.raises(ValueError, match="LIMIT must be an integer"):
        env_int("LIMIT", 5, 1, 10, {"LIMIT": "many"})


def test_checkpointing_security_invariant() -> None:
    assert CHECKPOINTING_ENABLED is False
