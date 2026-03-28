from pathlib import Path

from setuptools import find_packages


def test_package_discovery_includes_only_runtime_package():
    """Ensure packaging only includes the serving runtime package."""
    project_root = Path(__file__).resolve().parents[1]

    discovered = set(find_packages(where=str(project_root), include=["app*"]))

    assert discovered == {"app"}
