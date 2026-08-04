import pytest

from formosa.geomorphology.flowdir.graphs.graphs_py import _lines_intersect_v2

from test_distances import LINE_INTERSECTION_CASES


@pytest.mark.parametrize(
    ("l1a", "l1b", "l2a", "l2b", "expected"), LINE_INTERSECTION_CASES
)
def test_lines_intersect_v2(l1a, l1b, l2a, l2b, expected):
    assert _lines_intersect_v2(l1a, l1b, l2a, l2b) == expected


@pytest.mark.parametrize(
    ("l1a", "l1b", "l2a", "l2b", "expected"), LINE_INTERSECTION_CASES
)
def test_lines_intersect_v2_is_symmetric(l1a, l1b, l2a, l2b, expected):
    variants = [
        (l2a, l2b, l1a, l1b),
        (l1b, l1a, l2a, l2b),
        (l1a, l1b, l2b, l2a),
        (l1b, l1a, l2b, l2a),
    ]
    for args in variants:
        assert _lines_intersect_v2(*args) == expected
