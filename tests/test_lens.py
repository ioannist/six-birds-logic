import numpy as np

from emergent_logic.lens import fibers, is_definable_predicate, pushforward, validate_lens


def test_fibers_partition() -> None:
    f = np.array([0, 0, 1, 1, 2], dtype=int)
    grouped = fibers(f)
    assert len(grouped) == 3
    flat = np.concatenate(grouped).tolist()
    assert flat == [0, 1, 2, 3, 4]
    assert len(set(flat)) == len(flat)


def test_pushforward_preserves_mass() -> None:
    f = np.array([0, 0, 1, 1, 2], dtype=int)
    mu = np.array([0.1, 0.2, 0.05, 0.25, 0.4], dtype=float)
    nu = pushforward(mu, f)
    assert abs(float(nu.sum()) - float(mu.sum())) < 1e-12
    np.testing.assert_allclose(nu, np.array([0.3, 0.30, 0.4]), atol=1e-12)


def test_definability_pass_and_fail() -> None:
    f = np.array([0, 0, 1, 1, 2], dtype=int)
    p_ok = np.array([0, 0, 1, 1, 7], dtype=int)
    p_bad = np.array([0, 1, 1, 1, 7], dtype=int)
    assert is_definable_predicate(p_ok, f) is True
    assert is_definable_predicate(p_bad, f) is False


def test_validate_lens_failures() -> None:
    f_negative = np.array([0, -1, 1], dtype=int)
    assert validate_lens(f_negative, raise_on_fail=False) is False

    f = np.array([0, 0, 1, 1, 2], dtype=int)
    assert validate_lens(f, n_macro=2, raise_on_fail=False) is False
