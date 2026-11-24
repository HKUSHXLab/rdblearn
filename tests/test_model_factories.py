import pytest

from rdblearn.constants import TABICL_DEFAULT_CONFIG, TABPFN_DEFAULT_CONFIG
from rdblearn.estimator import RDBLearnClassifier
from rdblearn.utils import (
    TABPFN_VERSIONS,
    TFM_FACTORIES,
    make_tabicl_classifier,
    make_tabicl_regressor,
    make_tabpfn_classifier,
    make_tabpfn_regressor,
)


@pytest.mark.parametrize("version", TABPFN_VERSIONS)
def test_make_tabpfn_versions(version):
    pytest.importorskip("tabpfn")
    for est in (make_tabpfn_classifier(version, device="cpu"), make_tabpfn_regressor(version, device="cpu")):
        assert est.fit_mode == "fit_with_cache"
        assert est.device == "cpu"
        assert est.n_estimators == TABPFN_DEFAULT_CONFIG["n_estimators"]
    expected = {
        "v3.5": "tabpfn-v3.5-20260909.safetensors",
        "v3.5-fast": "tabpfn-v3.5-fast-20260909.safetensors",
    }.get(version)
    if expected is not None:
        assert str(make_tabpfn_classifier(version).model_path).endswith(expected)


def test_make_tabpfn_overrides_and_unknown_version():
    pytest.importorskip("tabpfn")
    clf = make_tabpfn_classifier("v3.5-fast", fit_mode="fit_preprocessors", n_estimators=4)
    assert (clf.fit_mode, clf.n_estimators) == ("fit_preprocessors", 4)
    with pytest.raises(ValueError):
        make_tabpfn_classifier("v4")


def test_make_tabicl_uses_v2_with_kv_cache():
    pytest.importorskip("tabicl")
    clf, reg = make_tabicl_classifier(device="cpu"), make_tabicl_regressor(device="cpu")
    assert clf.checkpoint_version.startswith("tabicl-classifier-v2")
    assert reg.checkpoint_version.startswith("tabicl-regressor-v2")
    for est in (clf, reg):
        assert est.kv_cache is True
        assert est.n_estimators == TABICL_DEFAULT_CONFIG["n_estimators"]
        assert est.device == "cpu"


def test_tfm_factories_cover_both_tasks():
    pytest.importorskip("tabpfn")
    pytest.importorskip("tabicl")
    for name, (make_clf, make_reg) in TFM_FACTORIES.items():
        assert make_clf(device="cpu") is not None, name
        assert make_reg(device="cpu") is not None, name


def test_class_cap_per_base_estimator():
    pytest.importorskip("tabpfn")
    pytest.importorskip("tabicl")
    cap = lambda est: RDBLearnClassifier(base_estimator=est)._base_estimator_max_classes()
    assert cap(make_tabpfn_classifier("v3.5", device="cpu")) == 160
    assert cap(make_tabpfn_classifier("v2.5", device="cpu")) == 10
    assert cap(make_tabicl_classifier(device="cpu")) is None
    assert cap(object()) == 10
