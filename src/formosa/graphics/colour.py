"""
Defines and transforms colour maps for terrain visualisation.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def truncate_colormap(
    cmap: LinearSegmentedColormap,
    minval: float = 0.0,
    maxval: float = 1.0,
    n: int = 256,
) -> LinearSegmentedColormap:
    """
    Truncates a colour map to a specified range.

    Parameters
    ----------
    cmap : LinearSegmentedColormap
        Source colour map to truncate.
    minval : float, optional
        Lower bound of the normalised range `[0, 1]`.
        - Default value is `0.0`.
    maxval : float, optional
        Upper bound of the normalised range `[0, 1]`.
        - Default value is `1.0`.
    n : int, optional
        Number of interpolation points.
        - Default point count is `256`.

    Returns
    -------
    new_cmap : LinearSegmentedColormap
        Truncated colour map.
    """
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


light_terrain_val: list[str] = [
    "#1C2E23",
    "#2B5E40",
    "#629456",
    "#ABBF7A",
    "#E3D7A8",
    "#F7F2E6",
]


def light_terrain(
    minval: float = 0.0,
    maxval: float = 1.0,
    N: int = 256,
    reverse: bool = False,
    **kwargs,
) -> LinearSegmentedColormap:
    """
    Loads a light terrain colour map.

    Parameters
    ----------
    minval : float, optional
        Lower truncation bound.
        - Default value is `0.0`.
    maxval : float, optional
        Upper truncation bound.
        - Default value is `1.0`.
    N : int, optional
        Number of colour levels.
        - Default level count is `256`.
    reverse : bool, optional
        Whether to reverse the colour sequence.
        - Default option is `False`.
    **kwargs
        Additional keyword arguments passed to
        `LinearSegmentedColormap.from_list`.

    Returns
    -------
    cmap : LinearSegmentedColormap
        Light terrain colour map.
    """
    cmap = LinearSegmentedColormap.from_list(
        "light_terrain_cmap",
        light_terrain_val if not reverse else light_terrain_val[::-1],
        N=N,
        **kwargs,
    )
    return truncate_colormap(cmap, minval, maxval, N)


dune_val: list[str] = ["#012E40", "#3D545C", "#8C7B62", "#CFA97E", "#EBD19D", "#F7F4E6"]


def dune(
    minval: float = 0.0,
    maxval: float = 1.0,
    N: int = 256,
    reverse: bool = False,
    alpha: bool = False,
    **kwargs,
) -> LinearSegmentedColormap:
    """
    Loads a dune earth-tone colour map.

    Parameters
    ----------
    minval : float, optional
        Lower truncation bound.
        - Default value is `0.0`.
    maxval : float, optional
        Upper truncation bound.
        - Default value is `1.0`.
    N : int, optional
        Number of colour levels.
        - Default level count is `256`.
    reverse : bool, optional
        Whether to reverse the colour sequence.
        - Default option is `False`.
    alpha : bool, optional
        Whether to include custom alpha transparency values.
        - Default option is `False`.
    **kwargs
        Additional keyword arguments passed to
        `LinearSegmentedColormap.from_list`.

    Returns
    -------
    cmap : LinearSegmentedColormap
        Dune colour map.
    """
    if alpha:
        dune_val_w_alpha = dune_val.copy()
        dune_val_w_alpha[0] += "FF"
        dune_val_w_alpha[-1] += "FF"
        dune_val_w_alpha[1] += "88"
        dune_val_w_alpha[-2] += "88"
        dune_val_w_alpha[2] += "11"
        dune_val_w_alpha[-3] += "11"
    else:
        dune_val_w_alpha = dune_val.copy()
    cmap = LinearSegmentedColormap.from_list(
        "dune_cmap",
        dune_val_w_alpha if not reverse else dune_val_w_alpha[::-1],
        N=N,
        **kwargs,
    )
    return truncate_colormap(cmap, minval, maxval, N)


mist_val: list[str] = ["#012F38", "#115A73", "#2F8EBD", "#78BEF0", "#BDD8FF", "#F0F3FC"]


def mist(
    minval: float = 0.0,
    maxval: float = 1.0,
    N: int = 256,
    reverse: bool = False,
    **kwargs,
) -> LinearSegmentedColormap:
    """
    Loads a mist blue-tone colour map.

    Parameters
    ----------
    minval : float, optional
        Lower truncation bound.
        - Default value is `0.0`.
    maxval : float, optional
        Upper truncation bound.
        - Default value is `1.0`.
    N : int, optional
        Number of colour levels.
        - Default level count is `256`.
    reverse : bool, optional
        Whether to reverse the colour sequence.
        - Default option is `False`.
    **kwargs
        Additional keyword arguments passed to
        `LinearSegmentedColormap.from_list`.

    Returns
    -------
    cmap : LinearSegmentedColormap
        Mist colour map.
    """
    cmap = LinearSegmentedColormap.from_list(
        "mist_cmap",
        mist_val if not reverse else mist_val[::-1],
        N=N,
        **kwargs,
    )
    return truncate_colormap(cmap, minval, maxval, N)


iceberg_val: list[str] = [
    "#112D38",
    "#325563",
    "#608999",
    "#9FB9BF",
    "#C8DBDE",
    "#EBF7F7",
]


def iceberg(
    minval: float = 0.0,
    maxval: float = 1.0,
    N: int = 256,
    reverse: bool = False,
    **kwargs,
) -> LinearSegmentedColormap:
    """
    Loads an iceberg pale-blue colour map.

    Parameters
    ----------
    minval : float, optional
        Lower truncation bound.
        - Default value is `0.0`.
    maxval : float, optional
        Upper truncation bound.
        - Default value is `1.0`.
    N : int, optional
        Number of colour levels.
        - Default level count is `256`.
    reverse : bool, optional
        Whether to reverse the colour sequence.
        - Default option is `False`.
    **kwargs
        Additional keyword arguments passed to
        `LinearSegmentedColormap.from_list`.

    Returns
    -------
    cmap : LinearSegmentedColormap
        Iceberg colour map.
    """
    cmap = LinearSegmentedColormap.from_list(
        "iceberg_cmap",
        iceberg_val if not reverse else mist_val[::-1],
        N=N,
        **kwargs,
    )
    return truncate_colormap(cmap, minval, maxval, N)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = np.linspace(0, 1, 256).reshape(1, -1)
    plt.imshow(data, aspect="auto", cmap=iceberg())
    plt.axis("off")
    plt.show()
