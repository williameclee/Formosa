"""
Tests implementation details of the Python triangulation backend.

This module covers Python-only triangulation helpers and local mesh
operations that are not part of the public backend-independent API.

Created: 2026-08-17, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
import pytest

from formosa.geomorphology.drainage.network import GraphTopologyError
from formosa.geomorphology.meshing._backends import triangulation_py as tri_py


def test_symbolic_infinite_face_uses_hull_visibility():
    vtxs = np.array([[0, 0], [2, 0], [1, -1], [1, 0], [1, 1]], dtype=np.int32)
    iinf = vtxs.shape[0]
    inf_facet = (iinf, 1, 0)

    assert tri_py.is_bad_facet(inf_facet, 2, vtxs, iinf)
    assert tri_py.is_bad_facet(inf_facet, 3, vtxs, iinf)
    assert not tri_py.is_bad_facet(inf_facet, 4, vtxs, iinf)


def test_symbolic_infinite_predicate_rejects_invalid_cases():
    vtxs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
    iinf = vtxs.shape[0]

    with pytest.raises(GraphTopologyError, match="cannot be inserted"):
        tri_py.is_bad_facet((iinf, 1, 0), iinf, vtxs, iinf)

    with pytest.raises(GraphTopologyError, match="more than 1 infinite"):
        tri_py.is_bad_facet((iinf, iinf, 0), 1, vtxs, iinf)
