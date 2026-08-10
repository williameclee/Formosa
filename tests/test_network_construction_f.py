"""
Tests flow-graph construction using the FORTRAN backend.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

from tests.core import *

import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.drainage.network.construction as constr_m
from formosa.geomorphology._native import network_construction as constr_f

from types import SimpleNamespace


def test_construct_flowgraph_fortran_returns_buffer_overflow_code():
    dirs = np.array([[1, 1, 0]], dtype=np.uint8, order="F")
    valids = np.ones((1, 3), dtype=bool, order="F")
    orders = np.ones((1, 3), dtype=np.int16, order="F")
    seeds = np.array([[True, False, False]], dtype=bool, order="F")
    indegs = np.array([[0, 1, 1]], dtype=np.int8, order="F")
    offsets = np.array([[0, 1], [0, 0]], dtype=np.int32, order="F")
    codes = np.array([1, 0], dtype=np.uint8, order="F")

    *_, err_code = constr_f.construct_flowgraph(
        dirs, valids, orders, seeds, indegs, offsets, codes, True, 1
    )

    assert err_code == 3


def test_construct_flowgraph_translates_fortran_error(monkeypatch: pytest.MonkeyPatch):
    def fake_construct(*args):
        return (
            0,
            0,
            np.zeros(1, dtype=np.int16),
            np.zeros((2, 2), dtype=np.int32),
            np.zeros((2, 1), dtype=np.int32),
            3,
        )

    monkeypatch.setattr(
        constr_m,
        "constr_f",
        SimpleNamespace(construct_flowgraph=fake_construct),
    )

    with pytest.raises(RuntimeError, match=r"construct_flowgraph.*error code 3"):
        constr_m.construct_flowgraph(
            np.array([[0]], dtype=np.uint8),
            orders=np.ones((1, 1), dtype=np.uint8),
            min_order=1,
            backend="fortran",
        )


@pytest.mark.parametrize(
    ("dirs", "dir_scheme", "exp_fgi", "exp_fgj", "should_warn"),
    [
        (
            [[3, 3, 3], [3, 3, 3], [1, 1, 0]],
            D8Directions(transform_codes=lambda x: x),
            [
                *(0, 1, np.nan, 0, 1, np.nan, 0, 1, np.nan),
                *(1, 2, np.nan, 1, 2, np.nan, 1, 2, np.nan),
                *(2, 2, np.nan, 2, 2, np.nan, 2, 2, np.nan),
            ],
            [
                *(0, 0, np.nan, 1, 1, np.nan, 2, 2, np.nan),
                *(0, 0, np.nan, 1, 1, np.nan, 2, 2, np.nan),
                *(0, 1, np.nan, 1, 2, np.nan, 2, 2, np.nan),
            ],
            False,
        ),
        (
            [[5, 1, 1], [5, 1, 1], [5, 1, 1]],
            D8Directions(transform_codes=lambda x: x),
            [
                *(0, 0, np.nan),
                *(1, 1, np.nan),
                *(2, 2, np.nan),
            ],
            [
                *(1, 2, np.nan),
                *(1, 2, np.nan),
                *(1, 2, np.nan),
            ],
            True,
        ),
    ],
)
def test_create_flowgraph_3x3(dirs, dir_scheme, exp_fgi, exp_fgj, should_warn):
    if should_warn:
        with pytest.warns(UserWarning, match="Some downstream indices out of bounds"):
            fg_i, fg_j = constr_m.create_flowgraph(
                np.array(dirs), dir_scheme=dir_scheme
            )
    else:
        fg_i, fg_j = constr_m.create_flowgraph(np.array(dirs), dir_scheme=dir_scheme)
    np.testing.assert_array_equal(fg_i, np.array(exp_fgi))
    np.testing.assert_array_equal(fg_j, np.array(exp_fgj))
