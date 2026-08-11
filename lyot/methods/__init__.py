"""Methods: how a task is taught, independent of what the task is.

A method turns one ``Task`` into one training ``Example``: it decides what to ask,
what to label, and how much latent budget to allow at a given curriculum stage.

Which states count as valid intermediates is a separate axis (``lyot.curriculum``),
injected at construction, so a run is specified by three independent choices::

    task: prosqa    method: coconut    curriculum: path     # upstream Coconut
    task: prosqa    method: coconut    curriculum: ring     # Coconut-BFS
    task: prosqa    method: atc        curriculum: path     # ATC

Everything is derived from the task rather than read from precomputed fields, so any
method runs on any task family supporting the queries its curriculum makes.
"""

from lyot.methods.base import Example, Method, build, get_method, register_method
from lyot.methods import baselines, coconut, atc, bridge  # noqa: F401  (registration side effect)

__all__ = ["Example", "Method", "build", "get_method", "register_method"]
