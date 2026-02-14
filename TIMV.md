* I have noticed that many machines few states, but a huge number of arcs.  Is
  there a "character class" optimization for FSTs that will make our algorithms
  significantly faster?  (The ideal thing would be to have all algorithms
  operate in a lifted way where the arcs don't need to be fully materialized,
  but might need to do some set logic (e.g., intersection, set subtraction), but
  then we can do what we need with the Lazy Precover DFA even more lazily!)

* **Precompute frontier powerset DFA and universality table.**

  Observation: In DirtyPeekaboo on PTB, the PowersetArena converges to ~44,700
  states by position ~70, after which each incremental step costs 0.1ms because
  every resume frontier state is immediately classified as QSTOP via the
  `fst_univ_cache`.  This happens because at the frontier (buf_len >= step_n),
  the NFA arcs are target-independent — any output is accepted.  So universality
  depends only on the set of FST states, not the target content.

  Idea: precompute this entire structure ahead of time (once per FST, no target
  needed):

  1. Build the **powerset DFA of the FST over the input alphabet**, ignoring the
     output tape (standard subset construction where all output symbols are
     treated as "accept").  This enumerates all reachable sets of FST states.

  2. Compute **universality for every powerset state** via backward fixed-point:
     mark non-universal if non-final, missing arcs for some input symbol, or has
     an arc to a non-universal state.  Iterate to convergence.  O(|DFA| * |Sigma|).

  3. Store as a compact lookup table: powerset state ID -> 1 bit (universal or
     not).  ~44,700 states = ~5.5 KB for PTB.  Index via perfect hashing or
     direct integer IDs.

  4. Optionally, also precompute **per-output-symbol transitions** on the
     powerset DFA: for each state S and output symbol gamma, delta(S, gamma) =
     eps_closure({j : exists i in S with arc i->j output gamma}).  This turns
     the pre-frontier BFS into a chain of O(1) table lookups:
     S0 -> S1 = delta(S0, y1) -> S2 = delta(S1, y2) -> ... -> Sn, where
     is_universal[Sn] is a precomputed bit.  The 0-to-70 warm-up phase
     disappears entirely.

  Important caveat: the `fst_univ_cache` (and thus this precomputation) only
  applies to **pure frontier** powerset states — states where ALL NFA elements
  have buf_len == step_n + 1.  Mixed states (some elements still mid-target)
  have target-dependent universality because the pre-frontier elements' arcs
  depend on matching specific target symbols.  The code guards this explicitly
  (`all_frontier` check at peekaboo.rs:1202-1205).  Mixed states fall through
  to the full universality machinery (witness check, pos/neg caches, sub-BFS).
  For PTB at steady state, essentially all encountered states are pure-frontier,
  so precomputation covers ~100% of queries.  For other FSTs this fraction could
  be lower, requiring on-demand fallback.

  Risk: the full powerset DFA (without target filtering) could be larger than
  the target-specific one, since it includes states reachable via any output
  sequence.  For PTB (296 FST states) this is likely tractable.  For large BPE
  FSTs, may need a size bound — explore up to N states, fall back to on-demand
  BFS for the rest.

  The `fst_univ_cache` already survives across targets (not cleared in
  full_reset()), so this is partially implemented as a lazy version.  The
  proposal is to do it eagerly, ahead of time.