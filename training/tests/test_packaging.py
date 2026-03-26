from pathlib import Path

from setuptools import find_namespace_packages


def test_package_discovery_includes_runtime_packages():
    """Ensure packaging still exposes the runtime namespaces."""
    project_root = Path(__file__).resolve().parents[1]

    discovered = set(
        find_namespace_packages(
            where=str(project_root),
            include=["pre_train*", "post_train*", "my_gpt_model*", "tokenizer*"],
        )
    )

    assert discovered == {"pre_train", "post_train", "my_gpt_model", "tokenizer"}
