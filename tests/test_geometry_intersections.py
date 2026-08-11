"""
Verifies line-segment intersection parity across configured
backends.

Last modified: 2026-08-11, En-Chi Lee (williameclee@gmail.com)
"""

import pytest
import numpy as np

from formosa.utils import BACKENDS
import formosa.geomorphology.geometry.intersections as intx_m
from formosa.geomorphology.geometry.intersections import IntersectionKind


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("l1a", "l1b", "l2a", "l2b", "exp_flag"),
    [
        # Parallel lines
        ((0, 0), (0, 1), (1, 0), (1, 1), IntersectionKind.DISJOINT_SEGMENTS),
        ((0, 0), (3, 3), (1, 2), (0, 3), IntersectionKind.DISJOINT_SEGMENTS),
        ((0, 0), (1, 0), (2, 0), (3, 0), IntersectionKind.DISJOINT_SEGMENTS),
        # Sharing endpoints
        ((0, 0), (1, 0), (1, 0), (1, 1), IntersectionKind.ENDPOINT_CONTACT),
        ((0, 0), (1, 0), (1, 0), (2, 0), IntersectionKind.ENDPOINT_CONTACT),
        ((0, 0), (-2, -2), (3, 1), (-2, -2), IntersectionKind.ENDPOINT_CONTACT),
        # Crossing
        ((0, 0), (1, 1), (1, 0), (0, 1), IntersectionKind.INTERIOR_CROSSING),
        ((0, 0), (3, 3), (1, 2), (3, 0), IntersectionKind.INTERIOR_CROSSING),
        # Collinear overlapping lines
        ((0, 0), (0, 2), (0, 1), (0, 3), IntersectionKind.COLLINEAR_OVERLAP),
        ((0, 0), (4, 0), (1, 0), (3, 0), IntersectionKind.COLLINEAR_OVERLAP),
        ((0, 0), (2, 2), (1, 1), (3, 3), IntersectionKind.COLLINEAR_OVERLAP),
        # Collinear overlapping lines, sharing endpoints
        ((0, 0), (0, 2), (0, 1), (0, 2), IntersectionKind.COLLINEAR_OVERLAP),
        ((0, 0), (3, 3), (2, 2), (3, 3), IntersectionKind.COLLINEAR_OVERLAP),
        # Identical lines
        ((0, 0), (0, 1), (0, 0), (0, 1), IntersectionKind.IDENTICAL_SEGMENTS),
        ((2, 5), (4, 3), (4, 3), (2, 5), IntersectionKind.IDENTICAL_SEGMENTS),
        # T junction
        ((0, 0), (2, 0), (1, 1), (1, 0), IntersectionKind.T_JUNCTION),
        ((-1, -1), (3, 1), (1, 0), (5, 7), IntersectionKind.T_JUNCTION),
        # degenerate segment (some line is actually a point)
        ((0, 0), (0, 0), (0, 0), (1, 1), IntersectionKind.DEGENERATE_SEGMENT),
        ((0, 0), (0, 0), (1, 1), (1, 1), IntersectionKind.DEGENERATE_SEGMENT),
    ],
)
def test_intersection_parity(l1a, l1b, l2a, l2b, exp_flag, backend):
    flag: int = intx_m.lines_intersect_v2(l1a, l1b, l2a, l2b, backend=backend)
    assert flag == exp_flag
    # Flip the segments; result should be the same
    flag: int = intx_m.lines_intersect_v2(l2a, l2b, l1a, l1b, backend=backend)
    assert flag == exp_flag


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("function", "args", "expected"),
    [
        (intx_m.on_segment, ((0, 0), (2, 0), (1, 0)), True),
        (intx_m.bboxes_overlap, ((0, 0), (2, 2), (1, 1), (3, 3)), True),
        (intx_m.lines_intersect_v2, ((0, 0), (1, 1), (1, 0), (0, 1)), 1),
    ],
)
def test_public_wrappers_select_backend(backend, function, args, expected):
    result = function(*args, backend=backend)

    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("p1", "p2", "p3", "exp_det", "is_float"),
    [
        ((0, 0), (4, 0), (1, 3), 12, False),
        ((0, 0), (1, 3), (4, 0), -12, False),
        ((-3, 2), (1, 4), (5, 6), 0, False),
        ((2, 2), (2, 2), (8, -1), 0, False),
        ((-4, 7), (3, -2), (8, 6), 101, False),
        ((0.5, 0.25), (2.0, 0.25), (0.5, 2.25), 3.0, True),
        ((0.5, 0.25), (0.5, 2.25), (2.0, 0.25), -3.0, True),
        ((-0.5, 1.25), (0.75, 2.5), (2.0, 3.75), 0.0, True),
        ((0.1, 0.2), (1.4, -0.3), (-0.7, 2.1), 2.07, True),
        (
            np.array((0, 0), dtype=np.int32),
            np.array((50_000, 0), dtype=np.int32),
            np.array((0, 50_000), dtype=np.int32),
            2_500_000_000,
            False,
        ),  # Test overflow
    ],
)
def test_orient_v2(p1, p2, p3, exp_det, is_float, backend):
    det = intx_m.orient_v2(p1, p2, p3, backend=backend)
    if is_float:
        assert det == pytest.approx(exp_det, rel=1e-6, abs=1e-7)
        assert isinstance(det, float)
        assert intx_m.orient_v2(p1, p3, p2, backend=backend) == pytest.approx(
            -exp_det, rel=1e-6, abs=1e-7
        )
    else:
        assert det == exp_det
        assert intx_m.orient_v2(p1, p3, p2, backend=backend) == -exp_det


def test_orient_v2_python_resolves_nearly_collinear_integer_points():
    # The exact determinant is -1 despite both products being about 2.5e9.
    p1 = np.array((0, 0), dtype=np.int32)
    p2 = np.array((50_000, 49_999), dtype=np.int32)
    p3 = np.array((49_999, 49_998), dtype=np.int32)

    assert intx_m.orient_v2(p1, p2, p3, backend="python") == -1


@pytest.mark.parametrize("backend", BACKENDS)
def test_orient_v2_float_translation_invariance(backend):
    p1 = np.array((-1.25, 2.5))
    p2 = np.array((3.75, -0.5))
    p3 = np.array((2.0, 4.25))
    translation = np.array((10.125, -7.75))

    determinant = intx_m.orient_v2(p1, p2, p3, backend=backend)
    translated = intx_m.orient_v2(
        p1 + translation, p2 + translation, p3 + translation, backend=backend
    )

    assert translated == pytest.approx(determinant, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", [0.25, 2.5, -3.0])
def test_orient_v2_float_scaling_is_quadratic(backend, scale):
    p1 = np.array((-0.5, 1.25))
    p2 = np.array((2.0, -0.75))
    p3 = np.array((4.5, 3.0))

    determinant = intx_m.orient_v2(p1, p2, p3, backend=backend)
    scaled = intx_m.orient_v2(scale * p1, scale * p2, scale * p3, backend=backend)

    assert scaled == pytest.approx(scale * scale * determinant, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_orient_v2_python_preserves_fractional_determinant(dtype):
    p1 = np.array((0.125, -0.25), dtype=dtype)
    p2 = np.array((1.625, 0.5), dtype=dtype)
    p3 = np.array((-0.375, 2.0), dtype=dtype)

    determinant = intx_m.orient_v2(p1, p2, p3, backend="python")

    assert determinant == pytest.approx(3.75)
    assert isinstance(determinant, float)


def test_public_wrapper_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        intx_m.orient_v2((0, 0), (1, 0), (1, 1), backend="unknown")  # type: ignore


@pytest.mark.parametrize(
    ("point", "error"),
    [
        ((0, 1, 2), ValueError),
        (("x", "y"), TypeError),
        (np.array([1 + 2j, 3 + 4j]), TypeError),
    ],
)
def test_public_wrapper_validates_points(point, error):
    with pytest.raises(error):
        intx_m.orient_v2(point, (1, 0), (1, 1), backend="python")
