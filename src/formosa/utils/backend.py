"""
Helper definitions and functions for managing the 2 different
backends provided by this package.

Created by: 2026-08-09, En-Chi Lee (williameclee@gmail.com)
"""

from typing import Literal, Mapping, TypeAlias

BACKENDS = ("fortran", "python")
Backend: TypeAlias = Literal["fortran", "python"]

DEFAULT_ERROR_MAP: dict[int, tuple[type[Exception], str]] = {
    1: (ValueError, "invalid input"),
    2: (MemoryError, "unable to allocate backend workspace"),
    3: (RuntimeError, "array or index capacity exceeded"),
}


def raise_fortran_error(
    operation: str,
    err_code: int,
    errors: Mapping[int, tuple[type[Exception], str]] = DEFAULT_ERROR_MAP,
) -> None:
    """
    Raises the Python exception corresponding to a FORTRAN status
    code.

    The default project convention is:
      - 0: Success
      - 1: Invalid input
      - 2: Memory allocation failure
      - 3: Array or index overflow.
    A routine-specific mapping may refine exception types and text
    without changing those numeric meanings.

    Parameters
    ----------
    operation : str
        Name of the operation to display in the error message.
        This is typically the function name.
    err_code : int
        FORTRAN status code.
    errors : dict[int, tuple[Exception, str]], optional
        How `err_code` should be interpreted.
        Each code maps to a type of exception to raise and an error
        message.
        Default mapping is described as above.
    """
    if err_code == 0:
        return

    exception, msg = errors.get(
        int(err_code), (RuntimeError, "unknown backend failure")
    )
    raise exception(f"Fortran {operation} failed: {msg} (error code {err_code}).")
