import importlib.metadata
import tomllib
import unittest
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pinned_requirement(name):
    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    for dep in deps:
        req = Requirement(dep)
        if req.name == name:
            return req
    raise AssertionError(f"{name} is not listed in pyproject.toml dependencies")


class TestFastDFSDependency(unittest.TestCase):
    def test_pinned_fastdfs_is_at_least_1_3(self):
        req = _pinned_requirement("fastdfs")
        pinned = [Version(spec.version) for spec in req.specifier]
        self.assertTrue(pinned, "fastdfs must be pinned to a version")
        self.assertTrue(all(v >= Version("1.3") for v in pinned), str(req))

    def test_installed_fastdfs_satisfies_pin(self):
        req = _pinned_requirement("fastdfs")
        installed = importlib.metadata.version("fastdfs")
        self.assertIn(installed, req.specifier, f"installed fastdfs {installed} does not satisfy {req}")

    def test_fastdfs_provides_apis_used_by_rdblearn(self):
        # EncodeCategoryColumns first shipped in fastdfs 1.1; fastdfs 0.2.1 cannot import rdblearn.estimator.
        from fastdfs.transform import EncodeCategoryColumns  # noqa: F401
        from fastdfs.transform.encode_categorical import encode_series_with_label_encoder  # noqa: F401
        import rdblearn.estimator  # noqa: F401


if __name__ == "__main__":
    unittest.main()
