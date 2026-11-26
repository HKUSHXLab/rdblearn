import pytest

from rdblearn.constants import TABICL_DEFAULT_CONFIG, TABPFN_DEFAULT_CONFIG
from rdblearn.estimator import RDBLearnClassifier
import numpy as np
import pandas as pd
from sklearn.base import clone

from rdblearn.utils import (
    KUMO_TABULAR_SIZES,
    KumoTabularWrapperClassifier,
    KumoTabularWrapperRegressor,
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
    for size in KUMO_TABULAR_SIZES:
        clf, reg = (make(device="cpu") for make in TFM_FACTORIES[f"kumo-tabular-{size}"])
        assert isinstance(clf, KumoTabularWrapperClassifier) and clf.size == size
        assert isinstance(reg, KumoTabularWrapperRegressor) and reg.size == size
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
    assert cap(KumoTabularWrapperClassifier(device="cpu")) is None
    assert cap(object()) == 10


def _toy_frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.integers(0, 3, size=n)})
    return X, rng


def test_kumo_classifier_small_cpu():
    pytest.importorskip("sdm")
    X, rng = _toy_frame()
    y = np.where(X["a"] + 0.1 * rng.normal(size=len(X)) > 0, "pos", "neg")
    clf = KumoTabularWrapperClassifier(size="small", device="cpu", n_estimators=2)
    clf.fit(X.iloc[:80], y[:80])
    proba = clf.predict_proba(X.iloc[80:])
    assert proba.shape == (40, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-4)
    assert list(clf.classes_) == ["neg", "pos"]
    assert (clf.predict(X.iloc[80:]) == y[80:]).mean() > 0.8
    # The fitted context is reused: a second call must give the same result.
    np.testing.assert_allclose(clf.predict_proba(X.iloc[80:]), proba, rtol=1e-5)
    assert clone(clf).get_params() == clf.get_params()


def test_kumo_classifier_many_classes_single_head():
    pytest.importorskip("sdm")
    X, rng = _toy_frame(n=300)
    y = rng.integers(0, 13, size=len(X))
    clf = KumoTabularWrapperClassifier(size="small", device="cpu", n_estimators=1)
    clf.fit(X.iloc[:250], y[:250])
    assert clf.predict_proba(X.iloc[250:]).shape == (50, len(np.unique(y[:250])))


def test_kumo_regressor_small_cpu():
    pytest.importorskip("sdm")
    X, rng = _toy_frame()
    y = 3.0 * X["a"].to_numpy() + 0.1 * rng.normal(size=len(X))
    reg = KumoTabularWrapperRegressor(size="small", device="cpu", n_estimators=2)
    reg.fit(X.iloc[:80], y[:80])
    quantiles = reg.predict_quantiles(X.iloc[80:])
    assert quantiles.shape == (40, 999)
    np.testing.assert_allclose(reg.predict(X.iloc[80:]), quantiles[:, 499])
    assert np.corrcoef(reg.predict(X.iloc[80:]), y[80:])[0, 1] > 0.9
    reg.set_params(output_type="mean")
    np.testing.assert_allclose(reg.predict(X.iloc[80:]), quantiles.mean(axis=1))
