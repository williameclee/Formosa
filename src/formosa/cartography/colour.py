import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def truncate_colormap(
    cmap: LinearSegmentedColormap,
    minval: float = 0.0,
    maxval: float = 1.0,
    n: int = 256,
) -> LinearSegmentedColormap:
    """
    Truncate a colormap to a specified range.
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

    :param minval: Truncation minimum value
    :type minval: float
    :param maxval: Truncation maximum value
    :type maxval: float
    :param N: Number of colour levels
    :type N: int
    :param reverse: Whether to reverse the colour map
    :type reverse: bool
    :return: The light terrain colour map
    :rtype: LinearSegmentedColormap
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
    Loads a light terrain colour map.

    :param minval: Truncation minimum value
    :type minval: float
    :param maxval: Truncation maximum value
    :type maxval: float
    :param N: Number of colour levels
    :type N: int
    :param reverse: Whether to reverse the colour map
    :type reverse: bool
    :return: The light terrain colour map
    :rtype: LinearSegmentedColormap
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
    Loads a light terrain colour map.

    :param minval: Truncation minimum value
    :type minval: float
    :param maxval: Truncation maximum value
    :type maxval: float
    :param N: Number of colour levels
    :type N: int
    :param reverse: Whether to reverse the colour map
    :type reverse: bool
    :return: The light terrain colour map
    :rtype: LinearSegmentedColormap
    """
    cmap = LinearSegmentedColormap.from_list(
        "mist_cmap",
        mist_val if not reverse else mist_val[::-1],
        N=N,
        **kwargs,
    )
    return truncate_colormap(cmap, minval, maxval, N)

iceberg_val: list[str] = ["#112D38", "#325563", "#608999", "#9FB9BF", "#C8DBDE", "#EBF7F7"]


def iceberg(
    minval: float = 0.0,
    maxval: float = 1.0,
    N: int = 256,
    reverse: bool = False,
    **kwargs,
) -> LinearSegmentedColormap:
    """
    Loads a light terrain colour map.

    :param minval: Truncation minimum value
    :type minval: float
    :param maxval: Truncation maximum value
    :type maxval: float
    :param N: Number of colour levels
    :type N: int
    :param reverse: Whether to reverse the colour map
    :type reverse: bool
    :return: The light terrain colour map
    :rtype: LinearSegmentedColormap
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
