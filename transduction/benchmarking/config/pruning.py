"""Pruning configuration and presets for the benchmarking package."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PruningConfig:
    """Hyperparameters for pruning at each level of the computation.

    Attributes
    ----------
    Beam Search (CachedByteLM)
    --------------------------
    beam_k : int
        Number of BPE token beams to maintain in genlm ByteBeamState.
        Higher = more accurate but slower. Default: 5.
    beam_prune_threshold : float
        Drop beams with P < threshold * P_max. Default: 0.001.
    beam_cache_size : int
        Max cached beam states (LRU eviction). Default: 10000.

    Path Enumeration (BFS)
    ----------------------
    max_depth : int
        Maximum BFS path depth (bytes). Default: 30.
    max_paths : int
        Maximum accepting paths to enumerate per FSA. Default: 100.
    path_logp_threshold : float
        Prune paths with logp < best_logp + threshold. None = no pruning.

    Output Symbols
    --------------
    top_k_bytes : int
        Evaluate only top-K bytes by LM probability. 0 = all 256. Default: 0.
    include_common_ascii : bool
        Always include common ASCII (a-z, space, punctuation). Default: True.
    early_stop_mass : float
        Stop evaluating candidates once cumulative probability mass exceeds
        this threshold (0-1). None = evaluate all candidates. Default: None.

    LM-Guided Beam Search
    ---------------------
    use_lm_guided : bool
        Use LM-guided beam search instead of BFS + separate scoring.
    lm_beam_width : int
        Beam width for LM-guided search.
    lm_logp_floor : float
        Prune beam entries with logp below this absolute floor.
    use_unified : bool
        Use unified beam search over shared peekaboo DFA (requires
        IncrementalPeekaboo).
    """

    # Beam search
    beam_k: int = 5
    beam_prune_threshold: float = 0.001
    beam_cache_size: int = 10000

    # Path enumeration
    max_depth: int = 30
    max_paths: int = 100
    path_logp_threshold: Optional[float] = None

    # Output symbols
    top_k_bytes: int = 0
    include_common_ascii: bool = True
    early_stop_mass: Optional[float] = None

    # LM-guided beam search
    use_lm_guided: bool = False
    lm_beam_width: int = 32
    lm_logp_floor: float = -40.0
    use_unified: bool = False


# Preset configurations
CONFIGS = {
    "fast": PruningConfig(
        beam_k=3,
        beam_prune_threshold=0.01,
        beam_cache_size=1000,
        max_depth=15,
        max_paths=50,
        path_logp_threshold=-20.0,
        top_k_bytes=20,
        early_stop_mass=0.95,
    ),
    "balanced": PruningConfig(
        beam_k=5,
        beam_prune_threshold=0.001,
        beam_cache_size=10000,
        max_depth=30,
        max_paths=100,
    ),
    "smart": PruningConfig(
        # Moderate beam settings
        beam_k=5,
        beam_prune_threshold=0.001,
        beam_cache_size=10000,
        # Moderate path settings
        max_depth=30,
        max_paths=100,
        # Aggressive output pruning - only top 50 symbols by LM
        top_k_bytes=50,
        include_common_ascii=True,
        # Early stopping at 99% probability mass
        early_stop_mass=0.99,
    ),
    "exhaustive": PruningConfig(
        beam_k=10,
        beam_prune_threshold=0.0001,
        beam_cache_size=50000,
        max_depth=100,
        max_paths=500,
        path_logp_threshold=None,
        top_k_bytes=0,
    ),
    "beam": PruningConfig(
        beam_k=5,
        beam_prune_threshold=0.001,
        beam_cache_size=10000,
        max_depth=30,
        max_paths=100,
        top_k_bytes=50,
        early_stop_mass=0.99,
        use_lm_guided=True,
        lm_beam_width=32,
        lm_logp_floor=-40.0,
    ),
    "unified": PruningConfig(
        beam_k=5,
        beam_prune_threshold=0.001,
        beam_cache_size=10000,
        max_depth=30,
        max_paths=100,
        top_k_bytes=50,
        early_stop_mass=0.99,
        use_unified=True,
        lm_beam_width=64,
        lm_logp_floor=-40.0,
    ),
}
