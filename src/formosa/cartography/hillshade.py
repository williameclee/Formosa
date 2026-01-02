import numpy as np

import numpy.typing as npt


def hillshade(
    array: npt.NDArray[np.integer | np.floating],
    azimuth: int | float,
    altitude: int | float,
    method: str = "hard",
    zfactor: int | float = 1,
) -> npt.NDArray[np.floating]:
    """
    Compute hillshade from a 2D array.
    """
    array = array * zfactor
    azimuth = 180 - azimuth
    light_vector: npt.NDArray[np.floating] = np.array(
        [
            np.cos(np.deg2rad(altitude)) * np.cos(np.deg2rad(azimuth)),
            np.cos(np.deg2rad(altitude)) * np.sin(np.deg2rad(azimuth)),
            np.sin(np.deg2rad(altitude)),
        ]
    )

    dx, dy = np.gradient(array)
    normal_vector: npt.NDArray[np.integer | np.floating] = np.dstack(
        (-dx, -dy, np.ones_like(array))
    )
    normal_vector /= np.linalg.norm(normal_vector, axis=2, keepdims=True)
    intensity: npt.NDArray[np.floating] = np.sum(normal_vector * light_vector, axis=2)

    match method.lower():
        case "clamped" | "hard":
            intensity = np.clip(intensity, 0, 1)
        case "half":
            intensity = (intensity + 1) / 2
        case "soft" | "lambert":
            intensity = (intensity + 1) / 2
            intensity = intensity**2
        case _:
            raise ValueError(
                f"Hillshade method must be 'hard' ('clamped'), 'half', or 'soft' ('Lambert'), got '{method}' instead."
            )
    return intensity
