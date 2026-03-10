# Compiled Decomposition: Specializing FST Algorithms via Static Analysis

## Executive Summary

Our FST decomposition algorithms currently use a single general-purpose BFS
that handles all transducers uniformly. This proposal describes a framework
for **statically analyzing an FST's structure and generating specialized code
per region**, replacing generic BFS with a mixture of table lookups, sparse
linear algebra, and pre-materialized automata. For BPE-class transducers, this
eliminates runtime BFS entirely; for more complex transducers, it reduces cost
in proportion to structural regularity.

We already have a working prototype of this idea: `GeneralizedBeam` classifies
FST states as either "hub" (fast path: trie-mass scoring via sparse matvec) or
"non-hub" (slow path: generic particle expansion). This proposal generalizes
that binary classification into a richer taxonomy and addresses the gaps.

## Motivation

The core operation in transduced language model decoding is **decomposition**:
given a target prefix, compute the quotient and remainder languages of the FST.
Today, this is done by BFS over a lazily-constructed powerset DFA derived from
the precover NFA. Each BFS step involves:

1. Epsilon closure (NFA-level BFS)
2. Powerset construction (set hashing, interning)
3. Universality checking (potentially a nested BFS)
4. Quotient/remainder extraction

For BPE transducers, most of this work is unnecessary. The FST has a single
hub state, deterministic token paths, and trivial universality (IP-universal
everywhere). The `CharacterBeam` algorithm already exploits this by replacing
the entire BFS with a precomputed trie and a single sparse matrix-vector
multiply per decoding step. But `CharacterBeam` is hand-written for the
BPE-specific structure; it doesn't generalize.

The question is: **can we automatically detect exploitable structure in
arbitrary FSTs and generate specialized code accordingly?**

## Approach: Region Decomposition

### Phase 1: Static FST Classification

Analyze each FST state along several dimensions:

| Property              | Values                    | Impact                              |
|-----------------------|---------------------------|-------------------------------------|
| IP-universal          | yes / no                  | Universality check: trivial vs. BFS |
| Final (accepting)     | yes / no                  | Triggers Q/R extraction             |
| Output-deterministic  | yes / no                  | Powerset stays size 1               |
| Epsilon structure     | none / chain / branching  | Closure cost                        |
| Self-loop hub         | yes / no                  | Token boundaries are free           |
| Bounded fan-out       | yes (degree k) / no       | Sub-DFA materializable              |

Most of these properties are already computed by our existing infrastructure
(IP-universality fixpoint, determinism checks, hub vocab BFS).

### Phase 2: Region Identification

Group states into connected **regions** of uniform type. Each region gets a
specialized computation strategy:

**Hub Region** (already implemented).
IP-universal + final + output-deterministic, all paths return to the hub.
Precompute an `OutputTrie`; scoring reduces to a sparse matvec. No runtime BFS.
Cost: O(|source vocabulary|) per decoding step. BPE transducers are entirely
one hub region.

**Deterministic Corridor.**
A chain of states where each has exactly one non-epsilon arc per source symbol
and the powerset never grows beyond size 1. No set operations, no universality
checks. Compiled to a flat lookup table. When a corridor exits to a hub, fuse
the corridor traversal into the hub's trie (this is already what `OutputTrie`
does implicitly for BPE).

**Bounded Diamond.**
Epsilon-branching that reconverges within bounded depth. The powerset grows but
stays bounded (max size k). Pre-materialize the sub-DFA offline so that
runtime transitions are table lookups. The Rust `PowersetArena` already
fast-paths size-1 powerset sets; this extends that to small fixed-size sets.
This is the highest-value gap for non-BPE tokenizers (SentencePiece unigram,
WordPiece) where bounded ambiguity currently receives the fully general
treatment.

**Universal Plateau.**
A subgraph where every DFA state is IP-universal (but not necessarily a hub).
Skip universality checking entirely within the region. The BFS still runs but
with one fewer expensive operation per state.

**Wild Region** (general case, current fallback).
Unbounded non-determinism, mixed universality. Full BFS with universality
sub-BFS, as in Peekaboo / DirtyPeekaboo today.

### Phase 3: Vertical Fusion

Beyond grouping states horizontally, fuse sequences of computation steps:

- **Corridor collapse**: A k-step deterministic chain becomes a single table
  entry, eliminating k-1 intermediate BFS iterations.
- **Universality propagation**: Precompute the entire "universal frontier" so
  the BFS can skip interior states of a universal subgraph.
- **Preimage short-circuit**: Precompute which (FST state, buffer length) pairs
  are guaranteed to be preimage states, enabling early termination.

### Phase 4: Vectorization

The trie-mass computation is already vectorized (sparse matvec via PyTorch).
Two further opportunities:

- **Batch across hypotheses**: Multiple hub hypotheses at the same hub share
  the same trie and reachability matrix. Stack their log-probability vectors
  into a matrix and perform a single sparse matrix-matrix multiply instead of
  K independent matvecs.
- **Batch across output symbols**: The per-child scoring loop can be expressed
  as a gather + logsumexp over a precomputed parent-to-children sparse index,
  converting the inner loop to matrix operations.

Both of these are natural targets for JAX/XLA compilation, enabling GPU
acceleration of the scoring pass.

## What Exists Today vs. What's New

| Capability                        | Status          | Effort   |
|-----------------------------------|-----------------|----------|
| Hub detection + trie-mass scoring | Implemented     | Done     |
| IP-universality computation       | Implemented     | Done     |
| Deterministic corridor detection  | Implicit in trie| Low      |
| Bounded diamond materialization   | Not implemented | Medium   |
| Universal plateau optimization    | Not implemented | Low      |
| Batched trie-mass (matmul)        | Not implemented | Low      |
| Static region analyzer            | Not implemented | Medium   |
| Rust compiled decomposer          | Not implemented | High     |
| JAX/XLA compilation               | Not implemented | High     |

## Proposed Implementation Plan

**Near-term (low-hanging fruit):**
1. Batched trie-mass: stack hub hypotheses into a matrix multiply. Direct
   speedup for large beam widths with minimal code change.
2. Static region analyzer: classify FST states and report region structure.
   Useful as a diagnostic even before code generation.

**Medium-term (highest impact):**
3. Bounded diamond pre-materialization in Rust. This directly addresses the
   performance gap for non-BPE tokenizers.
4. Universal plateau fast path: skip universality checks in certified regions.

**Longer-term (ambitious):**
5. Compiled decomposer: `analyze_fst(fst) -> CompiledDecomposer` in Rust that
   returns an object with per-region specialized `step()` methods.
6. JAX/XLA backend for the scoring pass, enabling GPU-accelerated decoding.

## Expected Impact

For BPE transducers, the current system is already close to optimal (the hub
region covers the entire FST). The main gains are:

- **Non-BPE tokenizers**: Bounded diamond materialization could reduce
  per-step cost from O(BFS exploration) to O(table lookup), potentially a
  10-100x speedup for the decomposition component.
- **Large beam widths**: Batched trie-mass replaces K independent matvecs with
  one matmul, giving up to K-fold speedup in the scoring pass.
- **General FSTs**: The region decomposition framework provides a principled
  way to exploit partial structure, smoothly interpolating between the
  fully-specialized (BPE/CharacterBeam) and fully-general (Peekaboo) cases.

The compiled decomposition idea also opens the door to **offline FST
compilation** as a deployment artifact: ship a pre-analyzed, pre-specialized
decomposer alongside the model, so that inference-time cost depends on the
FST's structural complexity rather than its raw size.
