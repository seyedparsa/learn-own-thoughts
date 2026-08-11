"""Tasks: what problem is being posed, independent of how it is taught.

A task owns its own data -- the graph, the question, the answer -- and can answer
structural queries about itself (how many hops, which nodes sit at depth d). It knows
nothing about curricula, latent tokens, or supervision: those live in ``lyot.methods``.

The split matters because the two vary independently. "Coconut on golden-path labels"
and "Coconut on BFS rings" were previously two *data files* shipped by upstream, even
though both are derived from the same graph -- verified: neighbor_k is recoverable
from (edges, root, target) for 15461/15461 rbs samples and 3000/3000 eos ball samples.
Deriving it in code turns a dataset swap into a one-word method choice, and lets a new
graph family drop in without regenerating anything per method.
"""

from lyot.tasks.base import Task, TaskSet, load_tasks
from lyot.tasks.reachability import ReachabilityTask, generate

__all__ = ["Task", "TaskSet", "load_tasks", "generate", "ReachabilityTask"]
