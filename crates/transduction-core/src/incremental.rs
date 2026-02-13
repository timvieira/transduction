//! Dirty-state incremental decomposition.
//!
//! `DirtyDecomp` persists the entire DFA structure (arena + per-state arcs +
//! state classification) across consecutive decomposition calls:
//!
//! - `decompose_dirty`: on prefix extension, only re-expands "dirty" states
//!   (whose NFA set contains frontier elements) and "border" states (clean
//!   states with arcs to dirty states). Clean states copy cached arcs.
//!
//! Requires **stable NFA state packing** via a fixed stride that is large
//! enough for all future target lengths.
//!
//! Caches are only reused when the new target is a strict prefix extension of
//! the previous target. If the target changes in any other way (different
//! symbols, shorter, etc.), all caches are rebuilt from scratch.

use crate::decompose::{FsaResult, ProfileStats, UniversalityFilter};
use crate::fst::Fst;
use crate::powerset::PowersetArena;
use crate::precover::PrecoverNFA;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::Instant;

// Re-import for arcs_buf type
type ArcsBuf = FxHashMap<u32, Vec<u64>>;

/// Evict stale eps_cache entries for a prefix extension.
/// An entry is stale if:
/// - Its key (NFA state) has buf_pos >= common_prefix_len (frontier state whose arcs change)
/// - Its value (closure result) contains any state with buf_pos >= common_prefix_len
///   (checked via stored max_buf_pos, avoiding full value scan)
fn evict_stale_eps_cache(
    cache: &mut FxHashMap<u64, (Rc<Vec<u64>>, u32)>,
    stride: u64,
    common_prefix_len: u32,
) {
    cache.retain(|&key, (_value, max_buf_pos)| {
        let buf_pos = (key % stride) as u32;
        buf_pos < common_prefix_len && *max_buf_pos < common_prefix_len
    });
}

/// Compute the length of the common prefix between two slices.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

// ---------------------------------------------------------------------------
// DirtyDecomp: true dirty-state incremental BFS
// ---------------------------------------------------------------------------

// State status constants for DirtyDecomp
const STATUS_NEW: u8 = 0;       // needs full expansion
const STATUS_INTERIOR: u8 = 1;  // non-final, expanded (has cached arcs)
const STATUS_QSTOP: u8 = 2;     // universal final, no outgoing arcs
const STATUS_RSTOP: u8 = 3;     // non-universal final, expanded (has cached arcs)

/// Lightweight result from `decompose_dirty()` — stats only.
/// Arc materialization happens on demand via `materialize_quotient()`/`materialize_remainder()`.
pub struct DirtyDecompUpdate {
    pub stats: ProfileStats,
}

/// Dirty-state incremental decomposition.
///
/// Persists the entire DFA structure (arena + per-state arcs + state
/// classification) across calls. On prefix extension, only "dirty" states
/// (whose NFA set contains frontier elements) and "border" states (clean
/// states with arcs to dirty states) are re-expanded. Clean states copy
/// their cached arcs, skipping compute_arcs and intern entirely.
///
/// `decompose_dirty()` returns a lightweight `DirtyDecompUpdate` (stats only,
/// no arc arrays). Arc materialization is deferred to `materialize_quotient()`
/// and `materialize_remainder()`, which are called only when Q/R FSAs are needed.
pub struct DirtyDecomp {
    arena: PowersetArena,
    filter: Option<UniversalityFilter>,
    eps_cache: FxHashMap<u64, (Rc<Vec<u64>>, u32)>,

    /// Per-state outgoing arcs: arcs_from[sid] = [(label, dest), ...]
    arcs_from: Vec<Vec<(u32, u32)>>,
    /// Per-state status classification.
    state_status: Vec<u8>,

    stride: u64,
    prev_target: Vec<u32>,

    /// Cache: sorted FST state set -> universality result.
    /// For DFA states where ALL NFA elements are at buf_pos == target_len
    /// (pure frontier), the universality sub-BFS only explores states at
    /// buf_pos == target_len, making the result purely FST-topology-dependent
    /// and target-independent. This cache never needs eviction.
    fst_univ_cache: FxHashMap<Vec<u32>, bool>,

    /// State IDs visited in DFS order from the last decompose_dirty() call.
    reachable: Vec<u32>,
    /// Dense bitvec: reachable_flags[sid] = true iff sid is in `reachable`.
    /// Used as the DFS visited set and for scanning only previously-reachable
    /// states during init.
    reachable_flags: Vec<bool>,
    /// Persistent buffer for dirty/border marking (avoids per-call allocation).
    needs_reexpand: Vec<bool>,
    /// max_bufpos[sid] = max buf_pos in arena.sets[sid]. Used for O(1) dirty
    /// marking instead of scanning the full NFA set.
    max_bufpos: Vec<u32>,
    /// Reverse arc index: reverse_arcs[dst] = [src1, src2, ...]
    /// Used for O(dirty × in_degree) border detection instead of O(reachable × avg_arcs).
    reverse_arcs: Vec<Vec<u32>>,
    /// Results from last decompose_dirty(), stored for lazy materialization.
    last_start: u32,
    last_q_stop: Vec<u32>,
    last_r_stop: Vec<u32>,
    /// Whether materialize_bfs() has been called since the last decompose_dirty().
    materialized: bool,
    /// Persistent visited buffer for materialize_bfs (avoids per-call allocation).
    materialize_visited: Vec<bool>,
    /// Track which entries were set in materialize_visited for sparse clearing.
    materialize_visited_list: Vec<u32>,
}

impl DirtyDecomp {
    /// Create a new DirtyDecomp with a fixed stride.
    /// stride should be large enough for all future target lengths
    /// (e.g., max_target_len + 1).
    pub fn new(_ip_univ: &[bool], stride: u64) -> Self {
        DirtyDecomp {
            arena: PowersetArena::new(),
            filter: None,
            eps_cache: FxHashMap::default(),
            arcs_from: Vec::new(),
            state_status: Vec::new(),
            stride,
            prev_target: Vec::new(),
            fst_univ_cache: FxHashMap::default(),
            reachable: Vec::new(),
            reachable_flags: Vec::new(),
            needs_reexpand: Vec::new(),
            max_bufpos: Vec::new(),
            reverse_arcs: Vec::new(),
            last_start: 0,
            last_q_stop: Vec::new(),
            last_r_stop: Vec::new(),
            materialized: false,
            materialize_visited: Vec::new(),
            materialize_visited_list: Vec::new(),
        }
    }

    /// Ensure all per-state arrays are sized to cover `needed` entries,
    /// populating max_bufpos for any newly-added arena states.
    fn ensure_capacity(&mut self, needed: usize) {
        let old_len = self.arcs_from.len();
        if needed > old_len {
            self.arcs_from.resize_with(needed, Vec::new);
            self.state_status.resize(needed, STATUS_NEW);
            self.reverse_arcs.resize_with(needed, Vec::new);
            // Populate max_bufpos for new arena entries
            self.max_bufpos.resize(needed, 0);
            for sid in old_len..needed {
                let nfa_set = &self.arena.sets[sid];
                let mbp = nfa_set.iter()
                    .map(|&e| (e % self.stride) as u32)
                    .max()
                    .unwrap_or(0);
                self.max_bufpos[sid] = mbp;
            }
        }
    }

    /// Remove `sid` from reverse_arcs of all its outgoing arc destinations,
    /// and clear its arcs_from (via std::mem::take).
    fn remove_outgoing_reverse_arcs(&mut self, sid: u32) {
        let arcs = std::mem::take(&mut self.arcs_from[sid as usize]);
        for &(_lbl, dst) in &arcs {
            let dst_usize = dst as usize;
            if dst_usize < self.reverse_arcs.len() {
                let ra = &mut self.reverse_arcs[dst_usize];
                if let Some(pos) = ra.iter().position(|&s| s == sid) {
                    ra.swap_remove(pos);
                }
            }
        }
    }

    /// Dirty-state decomposition: persists DFA structure across calls.
    /// On prefix extension, only re-expands dirty and border states via a
    /// local worklist. Clean states keep their cached arcs untouched.
    ///
    /// Returns a lightweight `DirtyDecompUpdate` with stats only (no arc arrays).
    /// Call `materialize_quotient()`/`materialize_remainder()` to get FSA results.
    pub fn decompose_dirty(&mut self, fst: &Fst, target: &[u32], ip_univ: &[bool]) -> DirtyDecompUpdate {
        let total_start = Instant::now();
        let mut stats = new_stats();
        let init_start = Instant::now();

        let target_len = target.len() as u32;
        let cplen = common_prefix_len(&self.prev_target, target);
        let is_extension = cplen == self.prev_target.len() && target.len() > self.prev_target.len();
        let has_prev = !self.prev_target.is_empty() && self.filter.is_some();

        // Collect dirty+border state IDs for worklist seeding
        let mut dirty_border: Vec<u32> = Vec::new();

        if is_extension && has_prev {
            let frontier = cplen as u32; // = prev_target_len

            // Evict stale eps_cache and filter entries
            evict_stale_eps_cache(&mut self.eps_cache, self.stride, frontier);
            if let Some(ref mut filter) = self.filter {
                filter.evict_frontier(ip_univ, target_len, self.stride, frontier);
            }

            // Ensure needs_reexpand covers arena (persistent buffer, all false between calls)
            let n = self.arena.len();
            if self.needs_reexpand.len() < n {
                self.needs_reexpand.resize(n, false);
            }

            // Step 1: Mark dirty states (max_bufpos >= frontier)
            for &sid in &self.reachable {
                let sid_usize = sid as usize;
                if sid_usize < self.max_bufpos.len() && self.max_bufpos[sid_usize] >= frontier {
                    self.needs_reexpand[sid_usize] = true;
                    dirty_border.push(sid);
                }
            }

            // Step 2: Mark border states using reverse_arcs [O(dirty × in_degree)]
            let dirty_count = dirty_border.len();
            for i in 0..dirty_count {
                let dirty_sid = dirty_border[i];
                let dirty_usize = dirty_sid as usize;
                if dirty_usize < self.reverse_arcs.len() {
                    for j in 0..self.reverse_arcs[dirty_usize].len() {
                        let src = self.reverse_arcs[dirty_usize][j];
                        let src_usize = src as usize;
                        if src_usize < n
                            && !self.needs_reexpand[src_usize]
                            && self.state_status[src_usize] != STATUS_NEW
                        {
                            self.needs_reexpand[src_usize] = true;
                            dirty_border.push(src);
                        }
                    }
                }
            }

            // Step 3: Reset dirty+border states
            for &sid in &dirty_border {
                let sid_usize = sid as usize;
                self.remove_outgoing_reverse_arcs(sid);
                self.state_status[sid_usize] = STATUS_NEW;
                self.arena.is_final[sid_usize] = false;
                self.needs_reexpand[sid_usize] = false;
            }
        } else if target != self.prev_target.as_slice() {
            // Non-extension change: full reset
            self.arena = PowersetArena::new();
            self.eps_cache.clear();
            self.filter = None;
            self.arcs_from.clear();
            self.state_status.clear();
            self.reachable.clear();
            self.reachable_flags.clear();
            self.needs_reexpand.clear();
            self.max_bufpos.clear();
            self.reverse_arcs.clear();
        }
        // If target == prev_target, reuse everything as-is

        if self.filter.is_none() {
            self.filter = Some(UniversalityFilter::with_ip_univ(ip_univ, target_len, self.stride));
        }

        let nfa = PrecoverNFA::with_stride_and_cache(
            fst,
            target,
            self.stride,
            std::mem::take(&mut self.eps_cache),
        );
        let num_source_symbols = fst.source_alphabet.len();

        // Compute start state
        let raw_starts = nfa.start_states();
        let mut init_closed = Vec::new();
        nfa.eps_closure_set(&raw_starts, &mut init_closed);
        let any_final = init_closed.iter().any(|&s| nfa.is_final(s));
        let start_id = self.arena.intern(init_closed, any_final);

        // Ensure per-state storage is allocated
        self.ensure_capacity(self.arena.len());

        // Ensure reachable_flags covers arena
        let needed_flags = self.arena.len().max(start_id as usize + 1);
        if self.reachable_flags.len() < needed_flags {
            self.reachable_flags.resize(needed_flags, false);
        }

        self.last_start = start_id;
        self.materialized = false;

        let mut worklist: VecDeque<u32> = VecDeque::new();

        // Add start_id to worklist if it needs expansion (first call or new start state)
        if self.state_status[start_id as usize] == STATUS_NEW {
            worklist.push_back(start_id);
            if !self.reachable_flags[start_id as usize] {
                self.reachable_flags[start_id as usize] = true;
                self.reachable.push(start_id);
            }
        }

        // Seed worklist with dirty+border states
        for &sid in &dirty_border {
            worklist.push_back(sid);
        }

        stats.init_ms = init_start.elapsed().as_secs_f64() * 1000.0;

        let bfs_start = Instant::now();

        // Take filter out of self to satisfy borrow checker
        // (is_universal needs &mut arena which is self.arena)
        let mut filter = self.filter.take().unwrap();
        let mut arcs_buf: ArcsBuf = FxHashMap::default();

        // Local worklist: only expand STATUS_NEW states
        while let Some(sid) = worklist.pop_front() {
            if self.state_status[sid as usize] != STATUS_NEW {
                continue; // already expanded (duplicate in worklist)
            }

            // Check finality and universality
            if self.arena.is_final[sid as usize] {
                let cache_key = {
                    let nfa_set = &self.arena.sets[sid as usize];
                    let all_frontier = nfa_set.iter()
                        .all(|&e| (e % self.stride) as u32 == target_len);
                    if all_frontier {
                        let mut fst_states: Vec<u32> = nfa_set.iter()
                            .map(|&e| (e / self.stride) as u32)
                            .collect();
                        fst_states.sort_unstable();
                        Some(fst_states)
                    } else {
                        None
                    }
                };

                let cached_result = cache_key.as_ref()
                    .and_then(|k| self.fst_univ_cache.get(k).copied());

                let is_uni = if let Some(cached) = cached_result {
                    stats.universal_calls += 1;
                    let nfa_set_clone = self.arena.sets[sid as usize].to_vec();
                    if cached {
                        filter.add_pos(&nfa_set_clone);
                    } else {
                        filter.add_neg(&nfa_set_clone);
                    }
                    cached
                } else {
                    let uni_start = Instant::now();
                    stats.universal_calls += 1;
                    let result = filter.is_universal(
                        sid, &nfa, &mut self.arena, num_source_symbols, &mut stats,
                    );
                    stats.universal_ms += uni_start.elapsed().as_secs_f64() * 1000.0;
                    if let Some(key) = cache_key {
                        self.fst_univ_cache.insert(key, result);
                    }
                    result
                };

                // is_universal may have grown the arena via sub-BFS intern
                self.ensure_capacity(self.arena.len());
                if self.arena.len() > self.reachable_flags.len() {
                    self.reachable_flags.resize(self.arena.len(), false);
                }

                if is_uni {
                    stats.universal_true += 1;
                    self.state_status[sid as usize] = STATUS_QSTOP;
                    continue;
                } else {
                    stats.universal_false += 1;
                }
            }

            let pset_size = self.arena.sets[sid as usize].len();
            if pset_size > stats.max_powerset_size {
                stats.max_powerset_size = pset_size;
            }

            let arcs_start = Instant::now();
            let all_arcs = nfa.compute_all_arcs_into(&self.arena.sets[sid as usize], &mut arcs_buf);
            stats.compute_arcs_ms += arcs_start.elapsed().as_secs_f64() * 1000.0;
            stats.compute_arcs_calls += 1;

            let mut cached = Vec::with_capacity(all_arcs.len());
            for (x, successor) in all_arcs {
                let intern_start = Instant::now();
                let succ_final = successor.iter().any(|&s| nfa.is_final(s));
                let dest_id = self.arena.intern(successor, succ_final);
                stats.intern_ms += intern_start.elapsed().as_secs_f64() * 1000.0;
                stats.intern_calls += 1;

                // Ensure per-state storage for dest_id
                let needed = dest_id as usize + 1;
                self.ensure_capacity(needed);
                if needed > self.reachable_flags.len() {
                    self.reachable_flags.resize(needed, false);
                }

                cached.push((x, dest_id));

                // Update reverse_arcs
                self.reverse_arcs[dest_id as usize].push(sid);

                // Add newly-created STATUS_NEW successors to worklist
                if self.state_status[dest_id as usize] == STATUS_NEW {
                    if !self.reachable_flags[dest_id as usize] {
                        self.reachable_flags[dest_id as usize] = true;
                        self.reachable.push(dest_id);
                        worklist.push_back(dest_id);
                    }
                }
            }

            // Cache arcs and classify state
            self.arcs_from[sid as usize] = cached;
            self.state_status[sid as usize] = if self.arena.is_final[sid as usize] {
                STATUS_RSTOP
            } else {
                STATUS_INTERIOR
            };
        }

        stats.bfs_ms = bfs_start.elapsed().as_secs_f64() * 1000.0;

        // Put filter back
        self.filter = Some(filter);

        let (hits, misses) = nfa.eps_cache_stats();
        stats.eps_cache_hits = hits;
        stats.eps_cache_misses = misses;

        // Save state for next call (reuse allocation)
        self.eps_cache = nfa.take_eps_cache();
        self.prev_target.clear();
        self.prev_target.extend_from_slice(target);

        // Stats (q_stops/r_stops deferred to materialization)
        stats.dfa_states = self.arena.len() as u32;
        stats.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        DirtyDecompUpdate { stats }
    }

    /// Full BFS from last_start to determine reachability, q_stop/r_stop lists,
    /// and clean up the reachable set (remove dead states). Called lazily by
    /// materialize_quotient()/materialize_remainder(). Idempotent per decompose call.
    fn materialize_bfs(&mut self) {
        if self.materialized {
            return;
        }
        self.materialized = true;

        let n = self.arena.len();

        // Clear previous visited entries (sparse clear)
        for &sid in &self.materialize_visited_list {
            if (sid as usize) < self.materialize_visited.len() {
                self.materialize_visited[sid as usize] = false;
            }
        }
        self.materialize_visited_list.clear();
        self.materialize_visited.resize(n, false);

        let mut bfs_queue: VecDeque<u32> = VecDeque::new();
        let mut new_reachable: Vec<u32> = Vec::new();

        if (self.last_start as usize) < n {
            self.materialize_visited[self.last_start as usize] = true;
            self.materialize_visited_list.push(self.last_start);
            bfs_queue.push_back(self.last_start);
            new_reachable.push(self.last_start);
        }

        self.last_q_stop.clear();
        self.last_r_stop.clear();

        while let Some(sid) = bfs_queue.pop_front() {
            match self.state_status[sid as usize] {
                STATUS_QSTOP => {
                    self.last_q_stop.push(sid);
                }
                STATUS_RSTOP => {
                    self.last_r_stop.push(sid);
                    for &(_lbl, dst) in &self.arcs_from[sid as usize] {
                        let dst_usize = dst as usize;
                        if dst_usize < n && !self.materialize_visited[dst_usize] {
                            self.materialize_visited[dst_usize] = true;
                            self.materialize_visited_list.push(dst);
                            new_reachable.push(dst);
                            bfs_queue.push_back(dst);
                        }
                    }
                }
                STATUS_INTERIOR => {
                    for &(_lbl, dst) in &self.arcs_from[sid as usize] {
                        let dst_usize = dst as usize;
                        if dst_usize < n && !self.materialize_visited[dst_usize] {
                            self.materialize_visited[dst_usize] = true;
                            self.materialize_visited_list.push(dst);
                            new_reachable.push(dst);
                            bfs_queue.push_back(dst);
                        }
                    }
                }
                _ => {} // STATUS_NEW shouldn't appear after decompose_dirty
            }
        }

        // Update reachable and reachable_flags (clean up dead states)
        for &sid in &self.reachable {
            if (sid as usize) < self.reachable_flags.len() {
                self.reachable_flags[sid as usize] = false;
            }
        }
        self.reachable = new_reachable;
        if self.reachable_flags.len() < n {
            self.reachable_flags.resize(n, false);
        }
        for &sid in &self.reachable {
            self.reachable_flags[sid as usize] = true;
        }
    }

    /// Materialize the quotient FSA from the last decompose_dirty() call.
    /// Triggers a full BFS (once) to determine reachability and q_stop/r_stop.
    pub fn materialize_quotient(&mut self) -> FsaResult {
        self.materialize_bfs();
        self.collect_arcs_trimmed(&self.last_q_stop.clone())
    }

    /// Materialize the remainder FSA from the last decompose_dirty() call.
    /// Triggers a full BFS (once) to determine reachability and q_stop/r_stop.
    pub fn materialize_remainder(&mut self) -> FsaResult {
        self.materialize_bfs();
        self.collect_arcs_trimmed(&self.last_r_stop.clone())
    }

    /// Per-symbol branching: produce Q/R for every output symbol extension.
    ///
    /// Uses a lightweight overlay per symbol that shares the base DFA's clean-state
    /// arcs. The base DirtyDecomp is NOT modified (arena may grow via interning,
    /// but arcs/status/reverse_arcs are untouched).
    pub fn decompose_next_all(
        &mut self,
        fst: &Fst,
        target: &[u32],
        ip_univ: &[bool],
        output_symbols: &[u32],
    ) -> FxHashMap<u32, (FsaResult, FsaResult)> {
        let target_len = target.len() as u32;
        let frontier = target_len; // = len(prev_target)

        // Ensure reachable set is clean (materialize_bfs removes dead states)
        self.materialize_bfs();

        // Step 1: Identify dirty+border states (shared across all symbols)
        let n = self.arena.len();
        if self.needs_reexpand.len() < n {
            self.needs_reexpand.resize(n, false);
        }

        let mut dirty_border: Vec<u32> = Vec::new();

        // Mark dirty states (max_bufpos >= frontier)
        for &sid in &self.reachable {
            let sid_usize = sid as usize;
            if sid_usize < self.max_bufpos.len() && self.max_bufpos[sid_usize] >= frontier {
                self.needs_reexpand[sid_usize] = true;
                dirty_border.push(sid);
            }
        }

        // Mark border states using reverse_arcs
        let dirty_count = dirty_border.len();
        for i in 0..dirty_count {
            let dirty_sid = dirty_border[i];
            let dirty_usize = dirty_sid as usize;
            if dirty_usize < self.reverse_arcs.len() {
                for j in 0..self.reverse_arcs[dirty_usize].len() {
                    let src = self.reverse_arcs[dirty_usize][j];
                    let src_usize = src as usize;
                    if src_usize < n
                        && !self.needs_reexpand[src_usize]
                        && self.state_status[src_usize] != STATUS_NEW
                    {
                        self.needs_reexpand[src_usize] = true;
                        dirty_border.push(src);
                    }
                }
            }
        }

        // Clear needs_reexpand
        for &sid in &dirty_border {
            self.needs_reexpand[sid as usize] = false;
        }

        let invalidated: FxHashSet<u32> = dirty_border.iter().copied().collect();

        // Base q_stops and r_stops (excluding invalidated)
        let base_q_stops: Vec<u32> = self.reachable.iter()
            .filter(|&&sid| !invalidated.contains(&sid)
                && self.state_status[sid as usize] == STATUS_QSTOP)
            .copied().collect();
        let base_r_stops: Vec<u32> = self.reachable.iter()
            .filter(|&&sid| !invalidated.contains(&sid)
                && self.state_status[sid as usize] == STATUS_RSTOP)
            .copied().collect();

        // Evict stale eps_cache into a working copy (don't modify main cache)
        let mut evicted_eps_cache = self.eps_cache.clone();
        evict_stale_eps_cache(&mut evicted_eps_cache, self.stride, frontier);

        let num_source_symbols = fst.source_alphabet.len();
        let mut results = FxHashMap::default();

        // Step 2: Per-symbol loop
        for &y in output_symbols {
            // Build target extension
            let mut target_ext: Vec<u32> = target.to_vec();
            target_ext.push(y);
            let target_ext_len = target_ext.len() as u32;

            // Create fresh filter for this symbol
            let mut filter = UniversalityFilter::with_ip_univ(
                ip_univ, target_ext_len, self.stride,
            );

            // Build PrecoverNFA with cloned evicted eps_cache
            let nfa = PrecoverNFA::with_stride_and_cache(
                fst, &target_ext, self.stride, evicted_eps_cache.clone(),
            );

            // Compute start state (intern into arena — append-only, harmless)
            let raw_starts = nfa.start_states();
            let mut init_closed = Vec::new();
            nfa.eps_closure_set(&raw_starts, &mut init_closed);
            let any_final = init_closed.iter().any(|&s| nfa.is_final(s));
            let start_id = self.arena.intern(init_closed, any_final);
            self.ensure_capacity(self.arena.len());

            // Overlay: stores arcs/status for dirty+border+new states only
            let mut overlay_arcs: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
            let mut overlay_status: FxHashMap<u32, u8> = FxHashMap::default();
            let mut overlay_reverse_add: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
            let mut overlay_reverse_remove: FxHashMap<u32, Vec<u32>> = FxHashMap::default();

            // Mark dirty+border as NEW in overlay, record reverse_arcs removals
            for &sid in &dirty_border {
                overlay_status.insert(sid, STATUS_NEW);
                overlay_arcs.insert(sid, Vec::new());
                for &(_lbl, dst) in &self.arcs_from[sid as usize] {
                    overlay_reverse_remove.entry(dst).or_default().push(sid);
                }
            }

            // BFS expansion
            let mut worklist: VecDeque<u32> = VecDeque::new();
            let mut expanding: FxHashSet<u32> = FxHashSet::default();

            for &sid in &dirty_border {
                worklist.push_back(sid);
                expanding.insert(sid);
            }

            // Seed with start if it needs expansion
            if !expanding.contains(&start_id) {
                let base_status = if (start_id as usize) < self.state_status.len() {
                    self.state_status[start_id as usize]
                } else {
                    STATUS_NEW
                };
                let status = *overlay_status.get(&start_id).unwrap_or(&base_status);
                if status == STATUS_NEW {
                    worklist.push_back(start_id);
                    expanding.insert(start_id);
                }
            }

            let mut arcs_buf: ArcsBuf = FxHashMap::default();
            let mut q_stops = base_q_stops.clone();
            let mut r_stops = base_r_stops.clone();

            while let Some(sid) = worklist.pop_front() {
                let base_status = if (sid as usize) < self.state_status.len() {
                    self.state_status[sid as usize]
                } else {
                    STATUS_NEW
                };
                let status = *overlay_status.get(&sid).unwrap_or(&base_status);

                if status != STATUS_NEW {
                    continue;
                }

                let sid_usize = sid as usize;

                // Compute finality from NFA (not arena.is_final, which may be
                // stale when the same powerset set was interned for a different
                // target length where is_final differed).
                let is_final_nfa = self.arena.sets[sid_usize].iter()
                    .any(|&s| nfa.is_final(s));

                // Check finality and universality
                if is_final_nfa {
                    let cache_key = {
                        let nfa_set = &self.arena.sets[sid_usize];
                        let all_frontier = nfa_set.iter()
                            .all(|&e| (e % self.stride) as u32 == target_ext_len);
                        if all_frontier {
                            let mut fst_states: Vec<u32> = nfa_set.iter()
                                .map(|&e| (e / self.stride) as u32)
                                .collect();
                            fst_states.sort_unstable();
                            Some(fst_states)
                        } else {
                            None
                        }
                    };

                    let cached_result = cache_key.as_ref()
                        .and_then(|k| self.fst_univ_cache.get(k).copied());

                    let is_uni = if let Some(cached) = cached_result {
                        let nfa_set_clone = self.arena.sets[sid_usize].to_vec();
                        if cached {
                            filter.add_pos(&nfa_set_clone);
                        } else {
                            filter.add_neg(&nfa_set_clone);
                        }
                        cached
                    } else {
                        let mut dummy_stats = new_stats();
                        let result = filter.is_universal(
                            sid, &nfa, &mut self.arena, num_source_symbols, &mut dummy_stats,
                        );
                        self.ensure_capacity(self.arena.len());
                        if let Some(key) = cache_key {
                            self.fst_univ_cache.insert(key, result);
                        }
                        result
                    };

                    if is_uni {
                        overlay_status.insert(sid, STATUS_QSTOP);
                        overlay_arcs.insert(sid, Vec::new());
                        q_stops.push(sid);
                        continue;
                    }
                }

                // Compute arcs
                let all_arcs = nfa.compute_all_arcs_into(
                    &self.arena.sets[sid_usize], &mut arcs_buf,
                );

                let mut cached = Vec::with_capacity(all_arcs.len());
                for (x, successor) in all_arcs {
                    let succ_final = successor.iter().any(|&s| nfa.is_final(s));
                    let dest_id = self.arena.intern(successor, succ_final);
                    self.ensure_capacity(dest_id as usize + 1);

                    cached.push((x, dest_id));
                    overlay_reverse_add.entry(dest_id).or_default().push(sid);

                    let dest_base_status = if (dest_id as usize) < self.state_status.len() {
                        self.state_status[dest_id as usize]
                    } else {
                        STATUS_NEW
                    };
                    let dest_status = *overlay_status.get(&dest_id).unwrap_or(&dest_base_status);
                    if dest_status == STATUS_NEW && !expanding.contains(&dest_id) {
                        worklist.push_back(dest_id);
                        expanding.insert(dest_id);
                    }
                }

                let final_status = if is_final_nfa {
                    r_stops.push(sid);
                    STATUS_RSTOP
                } else {
                    STATUS_INTERIOR
                };
                overlay_arcs.insert(sid, cached);
                overlay_status.insert(sid, final_status);
            }

            // Materialize trimmed Q/R using overlay view
            let q_fsa = self.collect_arcs_overlay_trimmed(
                start_id, &q_stops,
                &overlay_arcs, &overlay_status,
                &overlay_reverse_add, &overlay_reverse_remove,
            );
            let r_fsa = self.collect_arcs_overlay_trimmed(
                start_id, &r_stops,
                &overlay_arcs, &overlay_status,
                &overlay_reverse_add, &overlay_reverse_remove,
            );

            results.insert(y, (q_fsa, r_fsa));
        }

        results
    }

    /// Collect trimmed arcs using an overlay view on top of the base DFA.
    /// Forward BFS from start, backward BFS from stops, intersect.
    fn collect_arcs_overlay_trimmed(
        &self,
        start: u32,
        stops: &[u32],
        overlay_arcs: &FxHashMap<u32, Vec<(u32, u32)>>,
        overlay_status: &FxHashMap<u32, u8>,
        overlay_reverse_add: &FxHashMap<u32, Vec<u32>>,
        overlay_reverse_remove: &FxHashMap<u32, Vec<u32>>,
    ) -> FsaResult {
        let n = self.arena.len();
        let num_states = n as u32;

        if stops.is_empty() {
            return FsaResult {
                num_states, start: vec![], stop: vec![],
                arc_src: vec![], arc_lbl: vec![], arc_dst: vec![],
            };
        }

        // Forward BFS from start
        let mut fwd_reachable: FxHashSet<u32> = FxHashSet::default();
        let mut fwd_queue: VecDeque<u32> = VecDeque::new();
        fwd_reachable.insert(start);
        fwd_queue.push_back(start);

        while let Some(sid) = fwd_queue.pop_front() {
            let status = overlay_status.get(&sid)
                .copied()
                .unwrap_or(self.state_status[sid as usize]);

            if status == STATUS_QSTOP {
                continue;
            }

            let arcs: &[(u32, u32)] = overlay_arcs.get(&sid)
                .map(|v| v.as_slice())
                .unwrap_or(&self.arcs_from[sid as usize]);

            for &(_lbl, dst) in arcs {
                if fwd_reachable.insert(dst) {
                    fwd_queue.push_back(dst);
                }
            }
        }

        // Filter stops to forward-reachable
        let reachable_stops: Vec<u32> = stops.iter()
            .filter(|&&s| fwd_reachable.contains(&s))
            .copied().collect();

        if reachable_stops.is_empty() {
            return FsaResult {
                num_states, start: vec![], stop: vec![],
                arc_src: vec![], arc_lbl: vec![], arc_dst: vec![],
            };
        }

        // Backward BFS from stops through combined reverse arcs
        let mut backward: FxHashSet<u32> = FxHashSet::default();
        let mut bfs: VecDeque<u32> = VecDeque::new();
        for &sid in &reachable_stops {
            backward.insert(sid);
            bfs.push_back(sid);
        }

        while let Some(sid) = bfs.pop_front() {
            let sid_usize = sid as usize;

            // Base reverse_arcs minus overlay removals
            if sid_usize < self.reverse_arcs.len() {
                let remove = overlay_reverse_remove.get(&sid);
                for &src in &self.reverse_arcs[sid_usize] {
                    let is_removed = remove.map_or(false, |v| v.contains(&src));
                    if !is_removed && fwd_reachable.contains(&src) && !backward.contains(&src) {
                        backward.insert(src);
                        bfs.push_back(src);
                    }
                }
            }

            // Overlay reverse_arcs additions
            if let Some(additions) = overlay_reverse_add.get(&sid) {
                for &src in additions {
                    if fwd_reachable.contains(&src) && !backward.contains(&src) {
                        backward.insert(src);
                        bfs.push_back(src);
                    }
                }
            }
        }

        // Collect arcs
        let start_vec = if backward.contains(&start) {
            vec![start]
        } else {
            vec![]
        };

        let mut arc_src = Vec::new();
        let mut arc_lbl = Vec::new();
        let mut arc_dst = Vec::new();

        for &sid in &backward {
            let status = overlay_status.get(&sid)
                .copied()
                .unwrap_or(self.state_status[sid as usize]);

            if status == STATUS_QSTOP {
                continue;
            }

            let arcs: &[(u32, u32)] = overlay_arcs.get(&sid)
                .map(|v| v.as_slice())
                .unwrap_or(&self.arcs_from[sid as usize]);

            for &(l, d) in arcs {
                if backward.contains(&d) {
                    arc_src.push(sid);
                    arc_lbl.push(l);
                    arc_dst.push(d);
                }
            }
        }

        FsaResult {
            num_states,
            start: start_vec,
            stop: reachable_stops,
            arc_src,
            arc_lbl,
            arc_dst,
        }
    }

    /// Collect arcs trimmed by backward BFS from stop states.
    /// Only includes arcs where both src and dst can reach a stop state.
    fn collect_arcs_trimmed(&self, stops: &[u32]) -> FsaResult {
        let n = self.arena.len();
        let num_states = n as u32;

        if stops.is_empty() {
            return FsaResult {
                num_states,
                start: vec![],
                stop: vec![],
                arc_src: vec![],
                arc_lbl: vec![],
                arc_dst: vec![],
            };
        }

        // Backward BFS from stop states through reverse_arcs
        let mut backward = vec![false; n];
        let mut bfs: VecDeque<u32> = VecDeque::new();
        for &sid in stops {
            if (sid as usize) < n {
                backward[sid as usize] = true;
                bfs.push_back(sid);
            }
        }
        while let Some(sid) = bfs.pop_front() {
            let sid_usize = sid as usize;
            if sid_usize < self.reverse_arcs.len() {
                for &src in &self.reverse_arcs[sid_usize] {
                    let src_usize = src as usize;
                    if src_usize < n
                        && !backward[src_usize]
                        && self.reachable_flags[src_usize]
                    {
                        backward[src_usize] = true;
                        bfs.push_back(src);
                    }
                }
            }
        }

        // Filter start states
        let start = if (self.last_start as usize) < n && backward[self.last_start as usize] {
            vec![self.last_start]
        } else {
            vec![]
        };

        // Collect arcs where both src and dst are backward-reachable
        let mut arc_src = Vec::new();
        let mut arc_lbl = Vec::new();
        let mut arc_dst = Vec::new();
        for &sid in &self.reachable {
            let sid_usize = sid as usize;
            if backward[sid_usize] && self.state_status[sid_usize] != STATUS_QSTOP {
                for &(l, d) in &self.arcs_from[sid_usize] {
                    if (d as usize) < n && backward[d as usize] {
                        arc_src.push(sid);
                        arc_lbl.push(l);
                        arc_dst.push(d);
                    }
                }
            }
        }

        FsaResult {
            num_states,
            start,
            stop: stops.to_vec(),
            arc_src,
            arc_lbl,
            arc_dst,
        }
    }
}

fn new_stats() -> ProfileStats {
    ProfileStats {
        total_ms: 0.0,
        init_ms: 0.0,
        bfs_ms: 0.0,
        compute_arcs_ms: 0.0,
        compute_arcs_calls: 0,
        intern_ms: 0.0,
        intern_calls: 0,
        universal_ms: 0.0,
        universal_calls: 0,
        universal_true: 0,
        universal_false: 0,
        universal_sub_bfs_states: 0,
        universal_compute_arcs_calls: 0,
        dfa_states: 0,
        total_arcs: 0,
        q_stops: 0,
        r_stops: 0,
        max_powerset_size: 0,
        avg_powerset_size: 0.0,
        eps_cache_hits: 0,
        eps_cache_misses: 0,
    }
}
