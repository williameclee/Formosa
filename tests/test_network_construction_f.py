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


def test_construct_flowgraph_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]])
    valids = np.array([[T, F, T], [T, T, T], [T, T, T]])

    exp_orders = np.array([1, 1, 1, 2])
    exp_lengths = np.array([1, 2, 3, 1])
    exp_ijs = [
        np.array([[1, 1], [2, 1]]),
        np.array([[0, 2], [1, 2], [2, 2]]),
        np.array([[0, 0], [1, 0], [2, 0], [2, 1]]),
        np.array([[2, 1], [2, 2]]),
    ]
    arc_orders, vertex_ijs, arc_endpts = constr_m.construct_flowgraph(
        dirs, dir_scheme=dir_scheme, backend="fortran", min_order=1, valids=valids
    )
    arc_lengths = arc_endpts[:, 1] - arc_endpts[:, 0]

    np.testing.assert_array_equal(arc_orders, exp_orders)
    np.testing.assert_array_equal(arc_lengths, exp_lengths)

    for i, exp_ij in enumerate(exp_ijs):
        np.testing.assert_array_equal(
            vertex_ijs[arc_endpts[i, 0] : arc_endpts[i, 1] + 1], exp_ij
        )


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


def test_create_flowgraph_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8, order="F")

    expected_fg_i = np.array(
        [
            *(0, 1, np.nan, 0, 1, np.nan, 0, 1, np.nan),
            *(1, 2, np.nan, 1, 2, np.nan, 1, 2, np.nan),
            *(2, 2, np.nan, 2, 2, np.nan, 2, 2, np.nan),
        ]
    )
    expected_fg_j = np.array(
        [
            *(0, 0, np.nan, 1, 1, np.nan, 2, 2, np.nan),
            *(0, 0, np.nan, 1, 1, np.nan, 2, 2, np.nan),
            *(0, 1, np.nan, 1, 2, np.nan, 2, 2, np.nan),
        ]
    )
    fg_i, fg_j = constr_m.create_flowgraph(dirs, dir_scheme=dir_scheme)
    np.testing.assert_array_equal(fg_i, expected_fg_i)
    np.testing.assert_array_equal(fg_j, expected_fg_j)

    # Config 2
    dirs = np.array([[5, 1, 1], [5, 1, 1], [5, 1, 1]])
    expected_fg_i = np.array(
        [
            *(0, 0, np.nan),
            *(1, 1, np.nan),
            *(2, 2, np.nan),
        ]
    )
    expected_fg_j = np.array(
        [
            *(1, 2, np.nan),
            *(1, 2, np.nan),
            *(1, 2, np.nan),
        ]
    )
    with pytest.warns(UserWarning):
        fg_i, fg_j = constr_m.create_flowgraph(dirs, dir_scheme=dir_scheme)
    np.testing.assert_array_equal(fg_i, expected_fg_i)
    np.testing.assert_array_equal(fg_j, expected_fg_j)
