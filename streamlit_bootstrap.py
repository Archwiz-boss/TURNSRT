"""Start Streamlit while keeping this machine's mixed Python install usable.

The Conda environment contains the compatible pydantic/pydantic-core pair, while a
few speech packages live in the per-user site.  Import pydantic from Conda first,
then restore the user site so both groups of packages can coexist for this app.
"""

from __future__ import annotations

import os
import site
import sys


def _normalized(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _preload_conda_pydantic() -> None:
    user_sites = site.getusersitepackages()
    if isinstance(user_sites, str):
        user_sites = [user_sites]
    normalized_user_sites = {_normalized(path) for path in user_sites}

    removed_paths: list[tuple[int, str]] = []
    for index in range(len(sys.path) - 1, -1, -1):
        path = sys.path[index]
        if path and _normalized(path) in normalized_user_sites:
            removed_paths.append((index, sys.path.pop(index)))

    try:
        import pydantic  # noqa: F401
        import pydantic_core  # noqa: F401
    finally:
        for index, path in sorted(removed_paths):
            sys.path.insert(index, path)


def _check_dependencies() -> None:
    import numpy  # noqa: F401
    import opencc  # noqa: F401
    import plotly  # noqa: F401
    import streamlit  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import whisper  # noqa: F401


def main() -> None:
    _preload_conda_pydantic()

    if sys.argv[1:] == ["--check"]:
        _check_dependencies()
        return

    from streamlit.web import cli as streamlit_cli

    sys.argv[0] = "streamlit"
    streamlit_cli.main(prog_name="streamlit")


if __name__ == "__main__":
    main()
