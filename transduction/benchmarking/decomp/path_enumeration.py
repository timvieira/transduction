"""BFS path enumeration over FSAs with pruning."""

from collections import deque

import numpy as np

from transduction.fsa import EPSILON
from transduction.benchmarking.config.constants import NEG_INF
from transduction.benchmarking.config.pruning import CONFIGS

# line_profiler support: no-op decorator when not profiling
try:
    profile  # noqa: F821 - injected by kernprof
except NameError:
    def profile(func):
        return func


@profile
def enumerate_fsa_paths_bfs(
    fsa,
    lm_logp_next: np.ndarray,
    cfg=None,
    max_depth: int = None,
    max_paths: int = None,
):
    """Enumerate accepting paths in FSA using BFS with pruning.

    Yields (path_bytes, logp, final_state) for paths found.
    Uses BFS to ensure shorter paths are found first.

    Parameters
    ----------
    fsa : FSA
        The automaton to enumerate paths in.
    lm_logp_next : np.ndarray
        Log probabilities for each byte [256 or 257].
    cfg : PruningConfig
        Pruning parameters. If None, uses balanced preset.
    """
    if cfg is None:
        cfg = CONFIGS["balanced"]

    depth_limit = max_depth if max_depth is not None else cfg.max_depth
    path_limit = max_paths if max_paths is not None else cfg.max_paths
    logp_threshold = cfg.path_logp_threshold

    if not fsa.states or not fsa.start:
        return

    # (state, path, depth)
    frontier = deque([(s, [], 0) for s in fsa.start])
    paths_found = 0
    visited_at_depth = {}
    best_logp = NEG_INF

    while frontier and paths_found < path_limit:
        state, path, depth = frontier.popleft()

        if depth > depth_limit:
            continue

        # Skip if we've visited this state at this depth
        key = (state, len(path))
        if key in visited_at_depth:
            continue
        visited_at_depth[key] = True

        if state in fsa.stop:
            logp = sum(lm_logp_next[b] for b in path) if path else 0.0

            # Logp threshold pruning
            if logp_threshold is not None:
                if logp > best_logp:
                    best_logp = logp
                elif logp < best_logp + logp_threshold:
                    continue

            yield (path, logp, state)
            paths_found += 1
            continue

        # Expand state - epsilon transitions first (priority)
        for x, next_state in fsa.arcs(state):
            if x == EPSILON:
                frontier.appendleft((next_state, path, depth))
            else:
                try:
                    byte_val = int(x)
                    if byte_val < len(lm_logp_next):
                        frontier.append((next_state, path + [byte_val], depth + 1))
                except ValueError:
                    pass
