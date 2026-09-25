"""Each service must import with only its own optional-dependency group installed.

The development environment has every extra installed, so a module that imports a
package from another group (e.g. the monitor pulling in LightGBM) would pass every
other test and only fail inside its slim container. This test derives each group's
dependency closure from ``uv.lock`` and imports the service with everything else
blocked.
"""

from __future__ import annotations

import importlib.metadata as metadata
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVICES = {
    "producer": "stream",
    "feature_processor": "stream",
    "scorer": "stream",
    "predictor": "serving",
    "bootstrap": "training",
    "trainer": "training",
    "monitor": "monitoring",
}

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to read uv.lock")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _closure(extra: str) -> set[str]:
    exported = subprocess.run(
        ["uv", "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project",
         "--extra", extra, "--format", "requirements.txt"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    names = {"fraud-detection"}
    for line in exported.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            names.add(_canonical(re.split(r"[=<>~!;\[ ]", line, maxsplit=1)[0]))
    return names


def _blocked_modules(allowed: set[str]) -> set[str]:
    blocked = set()
    for module, distributions in metadata.packages_distributions().items():
        if not any(_canonical(d) in allowed for d in distributions):
            blocked.add(module)
    return blocked


@pytest.mark.parametrize(("service", "extra"), sorted(SERVICES.items()))
def test_service_imports_with_only_its_dependency_group(service: str, extra: str) -> None:
    blocked = _blocked_modules(_closure(extra))
    script = f"""
import importlib.abc, sys
BLOCKED = {sorted(blocked)!r}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"{{name}} is not in the '{extra}' dependency group")
        return None
sys.meta_path.insert(0, Guard())
import fraud_detection.{service}
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-2_000:]
