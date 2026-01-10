import pytest
import numpy as np

from formosa import D8Directions


def test_confluence_distance_2x2():
    from formosa.geomorphology.flowdir_f import flowdir as flowdir_f

    print("Available functions in flowdir_f:", dir(flowdir_f))

    directions = D8Directions(transform_codes=lambda x: x)
    offset_lookup = np.zeros((256, 2), dtype=np.int32)
    for code, (di, dj) in zip(directions.codes, directions.offsets):
        offset_lookup[code, :] = [di, dj]
        print(f"Code {code}: offset ({di:3d}, {dj:3d})")

    flowdirs = np.array([[3, 3], [1, 0]], dtype=np.uint8)
    x, y = np.meshgrid(
        np.arange(flowdirs.shape[1], dtype=np.float32),
        np.arange(flowdirs.shape[0], dtype=np.float32),
        indexing="xy",
    )

    common_kwargs = {
        "flowdirs": flowdirs.astype(np.uint8, order="F"),
        "x": x.astype(np.float32, order="F"),
        "y": y.astype(np.float32, order="F"),
        "offset_lookup": offset_lookup,
        "check_flag": True,
    }

    dists = flowdir_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 2.0)
    assert np.isclose(dists[1], 1.0)

    dists = flowdir_f.compute_confluence_dist([2, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 1.0)

    dists = flowdir_f.compute_confluence_dist([1, 1], [2, 1], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 0.0)

    dists = flowdir_f.compute_confluence_dist([1, 1], [1, 1], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 0.0)

    common_kwargs["flowdirs"] = np.array([[3, 3], [5, 1]], dtype=np.uint8, order="F")

    dists = flowdir_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 1.0)

    dists = flowdir_f.compute_confluence_dist([2, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 1.0)

    dists = flowdir_f.compute_confluence_dist([2, 1], [2, 2], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 0.0)

    common_kwargs["flowdirs"] = np.array([[2, 3], [1, 0]], dtype=np.uint8, order="F")

    dists = flowdir_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], np.sqrt(2))
    assert np.isclose(dists[1], 1.0)

    dists = flowdir_f.compute_confluence_dist([1, 1], [2, 1], **common_kwargs)
    assert np.isclose(dists[0], np.sqrt(2))
    assert np.isclose(dists[1], 1.0)

    dists = flowdir_f.compute_confluence_dist([1, 1], [2, 2], **common_kwargs)
    assert np.isclose(dists[0], np.sqrt(2))
    assert np.isclose(dists[1], 0.0)

    common_kwargs["flowdirs"] = np.array([[1, 0], [1, 7]], dtype=np.uint8, order="F")

    dists = flowdir_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 0.0)

    dists = flowdir_f.compute_confluence_dist([1, 1], [2, 1], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 2.0)

    dists = flowdir_f.compute_confluence_dist([2, 2], [2, 1], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 1.0)

    common_kwargs["flowdirs"] = np.array([[0, 5], [7, 7]], dtype=np.uint8, order="F")

    dists = flowdir_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 1.0)

    dists = flowdir_f.compute_confluence_dist([1, 1], [2, 2], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 2.0)

    dists = flowdir_f.compute_confluence_dist([2, 1], [2, 2], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 2.0)

    common_kwargs["flowdirs"] = np.array([[1, 0], [8, 7]], dtype=np.uint8, order="F")

    dists = flowdir_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 0.0)

    dists = flowdir_f.compute_confluence_dist([1, 1], [2, 1], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], np.sqrt(2))

    common_kwargs["flowdirs"] = np.array([[2, 3], [1, 0]], dtype=np.uint8, order="F")

    dists = flowdir_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], np.sqrt(2))
    assert np.isclose(dists[1], 1.0)

    dists = flowdir_f.compute_confluence_dist([1, 1], [2, 1], **common_kwargs)
    assert np.isclose(dists[0], np.sqrt(2))
    assert np.isclose(dists[1], 1.0)

    common_kwargs["flowdirs"] = np.array([[0, 5], [7, 6]], dtype=np.uint8, order="F")

    dists = flowdir_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 0)
    assert np.isclose(dists[1], 1.0)

    dists = flowdir_f.compute_confluence_dist([1, 1], [2, 2], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], np.sqrt(2))


def test_confluence_distance_3x3():
    from formosa.geomorphology.flowdir_f import flowdir as flowdir_f

    directions = D8Directions(transform_codes=lambda x: x)
    offset_lookup = np.zeros((256, 2), dtype=np.int32)
    for code, (di, dj) in zip(directions.codes, directions.offsets):
        offset_lookup[code, :] = [di, dj]
        print(f"Code {code}: offset ({di:3d}, {dj:3d})")

    flowdirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8)
    x, y = np.meshgrid(
        np.arange(flowdirs.shape[1], dtype=np.float32),
        np.arange(flowdirs.shape[0], dtype=np.float32),
        indexing="xy",
    )

    common_kwargs = {
        "flowdirs": flowdirs.astype(np.uint8, order="F"),
        "x": x.astype(np.float32, order="F"),
        "y": y.astype(np.float32, order="F"),
        "offset_lookup": offset_lookup,
        "check_flag": True,
    }

    dists = flowdir_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 3.0)
    assert np.isclose(dists[1], 2.0)
    dists = flowdir_f.compute_confluence_dist([1, 1], [1, 3], **common_kwargs)
    assert np.isclose(dists[0], 4.0)
    assert np.isclose(dists[1], 2.0)
    dists = flowdir_f.compute_confluence_dist([3, 1], [3, 3], **common_kwargs)
    assert np.isclose(dists[0], 2.0)
    assert np.isclose(dists[1], 0.0)

    common_kwargs["flowdirs"] = np.array(
        [[5, 1, 1], [5, 1, 1], [5, 1, 1]], dtype=np.uint8, order="F"
    )
    dists = flowdir_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 1.0)


if __name__ == "__main__":
    test_confluence_distance_2x2()
    test_confluence_distance_3x3()
