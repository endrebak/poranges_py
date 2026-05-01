from __future__ import annotations

import pytest
from hypothesis import settings


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--property-max-examples",
        action="store",
        default=None,
        help="Maximum number of Hypothesis examples to run for property tests.",
        type=int,
    )


def pytest_configure(config: pytest.Config) -> None:
    max_examples = config.getoption("--property-max-examples")
    if max_examples is None:
        return

    settings.register_profile(
        "pytest-property",
        max_examples=max_examples,
        deadline=None,
    )
    settings.load_profile("pytest-property")

