use pyo3::prelude::*;
use crate::fst::{Fst, compute_ip_universal_states};
use crate::{decompose, peekaboo, incremental, minimize};
use std::time::Instant;

/// Python-visible FST wrapper. Constructed once from Python arrays, then
/// passed to `decompose()` repeatedly.
#[pyclass]
pub struct RustFst {
    pub(crate) inner: Fst,
}

#[pymethods]
impl RustFst {
    #[new]
    #[pyo3(signature = (num_states, start_states, final_states, arc_src, arc_in, arc_out, arc_dst, source_alphabet))]
    fn new(
        num_states: u32,
        start_states: Vec<u32>,
        final_states: Vec<u32>,
        arc_src: Vec<u32>,
        arc_in: Vec<u32>,
        arc_out: Vec<u32>,
        arc_dst: Vec<u32>,
        source_alphabet: Vec<u32>,
    ) -> Self {
        let inner = Fst::new(
            num_states,
            start_states,
            &final_states,
            &arc_src,
            &arc_in,
            &arc_out,
            &arc_dst,
            source_alphabet,
        );
        RustFst { inner }
    }
}

/// Python-visible FSA result.
#[pyclass]
pub struct RustFsa {
    num_states: u32,
    start: Vec<u32>,
    stop: Vec<u32>,
    src: Vec<u32>,
    lbl: Vec<u32>,
    dst: Vec<u32>,
}

#[pymethods]
impl RustFsa {
    fn num_states(&self) -> u32 {
        self.num_states
    }

    fn start_states(&self) -> Vec<u32> {
        self.start.clone()
    }

    fn final_states(&self) -> Vec<u32> {
        self.stop.clone()
    }

    fn arcs(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        (self.src.clone(), self.lbl.clone(), self.dst.clone())
    }
}

/// Python-visible profiling stats.
#[pyclass]
pub struct RustProfileStats {
    #[pyo3(get)]
    pub total_ms: f64,
    #[pyo3(get)]
    pub init_ms: f64,
    #[pyo3(get)]
    pub bfs_ms: f64,
    #[pyo3(get)]
    pub compute_arcs_ms: f64,
    #[pyo3(get)]
    pub compute_arcs_calls: u64,
    #[pyo3(get)]
    pub intern_ms: f64,
    #[pyo3(get)]
    pub intern_calls: u64,
    #[pyo3(get)]
    pub universal_ms: f64,
    #[pyo3(get)]
    pub universal_calls: u64,
    #[pyo3(get)]
    pub universal_true: u64,
    #[pyo3(get)]
    pub universal_false: u64,
    #[pyo3(get)]
    pub universal_sub_bfs_states: u64,
    #[pyo3(get)]
    pub universal_compute_arcs_calls: u64,
    #[pyo3(get)]
    pub dfa_states: u32,
    #[pyo3(get)]
    pub total_arcs: u64,
    #[pyo3(get)]
    pub q_stops: u32,
    #[pyo3(get)]
    pub r_stops: u32,
    #[pyo3(get)]
    pub max_powerset_size: usize,
    #[pyo3(get)]
    pub avg_powerset_size: f64,
    #[pyo3(get)]
    pub eps_cache_hits: u64,
    #[pyo3(get)]
    pub eps_cache_misses: u64,
    #[pyo3(get)]
    pub minimize_ms: f64,
}

#[pymethods]
impl RustProfileStats {
    fn __repr__(&self) -> String {
        format!(
            "ProfileStats(\n\
             \x20 total={:.1}ms, init={:.1}ms, bfs={:.1}ms\n\
             \x20 compute_arcs: {:.1}ms ({} calls)\n\
             \x20 intern: {:.1}ms ({} calls)\n\
             \x20 universal: {:.1}ms ({} calls, {} true, {} false)\n\
             \x20   sub_bfs_states={}, sub_compute_arcs={}\n\
             \x20 dfa_states={}, arcs={}, q_stops={}, r_stops={}\n\
             \x20 powerset: max={}, avg={:.1}\n\
             \x20 eps_cache: {} hits, {} misses ({:.1}% hit rate)\n\
             )",
            self.total_ms, self.init_ms, self.bfs_ms,
            self.compute_arcs_ms, self.compute_arcs_calls,
            self.intern_ms, self.intern_calls,
            self.universal_ms, self.universal_calls, self.universal_true, self.universal_false,
            self.universal_sub_bfs_states, self.universal_compute_arcs_calls,
            self.dfa_states, self.total_arcs, self.q_stops, self.r_stops,
            self.max_powerset_size, self.avg_powerset_size,
            self.eps_cache_hits, self.eps_cache_misses,
            if self.eps_cache_hits + self.eps_cache_misses > 0 {
                self.eps_cache_hits as f64 / (self.eps_cache_hits + self.eps_cache_misses) as f64 * 100.0
            } else {
                0.0
            },
        )
    }
}

/// Python-visible decomposition result containing quotient and remainder.
#[pyclass]
pub struct DecompResult {
    #[pyo3(get)]
    quotient: Py<RustFsa>,
    #[pyo3(get)]
    remainder: Py<RustFsa>,
    #[pyo3(get)]
    stats: Py<RustProfileStats>,
}

/// Helper to wrap a Rust DecompResult into PyO3 objects.
fn wrap_decomp_result(py: Python<'_>, result: decompose::DecompResult, do_minimize: bool) -> PyResult<DecompResult> {
    let mut minimize_ms = 0.0f64;

    let q_fsa = if do_minimize {
        let t = Instant::now();
        let m = minimize::minimize(&result.quotient);
        minimize_ms += t.elapsed().as_secs_f64() * 1000.0;
        m
    } else {
        result.quotient
    };

    let r_fsa = if do_minimize {
        let t = Instant::now();
        let m = minimize::minimize(&result.remainder);
        minimize_ms += t.elapsed().as_secs_f64() * 1000.0;
        m
    } else {
        result.remainder
    };

    let q = Py::new(py, RustFsa {
        num_states: q_fsa.num_states, start: q_fsa.start, stop: q_fsa.stop,
        src: q_fsa.arc_src, lbl: q_fsa.arc_lbl, dst: q_fsa.arc_dst,
    })?;
    let r = Py::new(py, RustFsa {
        num_states: r_fsa.num_states, start: r_fsa.start, stop: r_fsa.stop,
        src: r_fsa.arc_src, lbl: r_fsa.arc_lbl, dst: r_fsa.arc_dst,
    })?;

    let s = &result.stats;
    let stats = Py::new(py, RustProfileStats {
        total_ms: s.total_ms, init_ms: s.init_ms, bfs_ms: s.bfs_ms,
        compute_arcs_ms: s.compute_arcs_ms, compute_arcs_calls: s.compute_arcs_calls,
        intern_ms: s.intern_ms, intern_calls: s.intern_calls,
        universal_ms: s.universal_ms, universal_calls: s.universal_calls,
        universal_true: s.universal_true, universal_false: s.universal_false,
        universal_sub_bfs_states: s.universal_sub_bfs_states,
        universal_compute_arcs_calls: s.universal_compute_arcs_calls,
        dfa_states: s.dfa_states, total_arcs: s.total_arcs,
        q_stops: s.q_stops, r_stops: s.r_stops,
        max_powerset_size: s.max_powerset_size, avg_powerset_size: s.avg_powerset_size,
        eps_cache_hits: s.eps_cache_hits, eps_cache_misses: s.eps_cache_misses,
        minimize_ms,
    })?;

    Ok(DecompResult { quotient: q, remainder: r, stats })
}

/// Perform the fused decomposition: given a Rust FST and a target sequence,
/// compute the quotient (Q) and remainder (R) automata.
#[pyfunction]
#[pyo3(signature = (fst, target, minimize=false))]
pub fn rust_decompose(py: Python<'_>, fst: &RustFst, target: Vec<u32>, minimize: bool) -> PyResult<DecompResult> {
    let result = decompose::decompose(&fst.inner, &target);
    wrap_decomp_result(py, result, minimize)
}

// ---------------------------------------------------------------------------
// Dirty-state incremental decomposition
// ---------------------------------------------------------------------------

/// Lightweight result from dirty-state decompose() — stats only, no arc arrays.
/// Arc materialization happens on demand via quotient()/remainder() on the decomposer.
#[pyclass]
pub struct DirtyStepResult {
    #[pyo3(get)]
    stats: Py<RustProfileStats>,
}

/// Python-visible dirty-state decomposition wrapper.
/// Persists entire DFA structure and only re-expands dirty/border states.
///
/// `decompose()` returns a lightweight `DirtyStepResult` (stats only).
/// Call `quotient()`/`remainder()` to materialize Q/R FSAs on demand.
#[pyclass(unsendable)]
pub struct RustDirtyStateDecomp {
    fst: Py<RustFst>,
    ip_univ: Vec<bool>,
    persistent: incremental::DirtyDecomp,
}

#[pymethods]
impl RustDirtyStateDecomp {
    #[new]
    #[pyo3(signature = (fst, stride=4096))]
    fn new(py: Python<'_>, fst: Py<RustFst>, stride: u64) -> Self {
        let ip_univ = compute_ip_universal_states(&fst.borrow(py).inner);
        let persistent = incremental::DirtyDecomp::new(&ip_univ, stride);
        RustDirtyStateDecomp { fst, ip_univ, persistent }
    }

    /// Run dirty-state DFS and return stats. Does NOT materialize arc arrays.
    /// Call quotient()/remainder() afterward to get FSAs on demand.
    #[pyo3(signature = (target, minimize=false))]
    fn decompose(&mut self, py: Python<'_>, target: Vec<u32>, minimize: bool) -> PyResult<DirtyStepResult> {
        let _ = minimize;
        let update = self.persistent.decompose_dirty(
            &self.fst.borrow(py).inner, &target, &self.ip_univ,
        );
        let s = update.stats;
        let stats = Py::new(py, RustProfileStats {
            total_ms: s.total_ms, init_ms: s.init_ms, bfs_ms: s.bfs_ms,
            compute_arcs_ms: s.compute_arcs_ms, compute_arcs_calls: s.compute_arcs_calls,
            intern_ms: s.intern_ms, intern_calls: s.intern_calls,
            universal_ms: s.universal_ms, universal_calls: s.universal_calls,
            universal_true: s.universal_true, universal_false: s.universal_false,
            universal_sub_bfs_states: s.universal_sub_bfs_states,
            universal_compute_arcs_calls: s.universal_compute_arcs_calls,
            dfa_states: s.dfa_states, total_arcs: s.total_arcs,
            q_stops: s.q_stops, r_stops: s.r_stops,
            max_powerset_size: s.max_powerset_size, avg_powerset_size: s.avg_powerset_size,
            eps_cache_hits: s.eps_cache_hits, eps_cache_misses: s.eps_cache_misses,
            minimize_ms: 0.0,
        })?;
        Ok(DirtyStepResult { stats })
    }

    /// Materialize the quotient FSA from the last decompose() call.
    #[pyo3(signature = (minimize=false))]
    fn quotient(&mut self, py: Python<'_>, minimize: bool) -> PyResult<Py<RustFsa>> {
        let fsa = self.persistent.materialize_quotient();
        let fsa = if minimize { minimize::minimize(&fsa) } else { fsa };
        Py::new(py, RustFsa {
            num_states: fsa.num_states, start: fsa.start, stop: fsa.stop,
            src: fsa.arc_src, lbl: fsa.arc_lbl, dst: fsa.arc_dst,
        })
    }

    /// Materialize the remainder FSA from the last decompose() call.
    #[pyo3(signature = (minimize=false))]
    fn remainder(&mut self, py: Python<'_>, minimize: bool) -> PyResult<Py<RustFsa>> {
        let fsa = self.persistent.materialize_remainder();
        let fsa = if minimize { minimize::minimize(&fsa) } else { fsa };
        Py::new(py, RustFsa {
            num_states: fsa.num_states, start: fsa.start, stop: fsa.stop,
            src: fsa.arc_src, lbl: fsa.arc_lbl, dst: fsa.arc_dst,
        })
    }

    /// Per-symbol branching: decompose for every next output symbol.
    /// Returns a DirtyNextResult with per-symbol (quotient, remainder) pairs.
    #[pyo3(signature = (target, output_symbols))]
    fn decompose_next(
        &mut self, py: Python<'_>, target: Vec<u32>, output_symbols: Vec<u32>,
    ) -> PyResult<DirtyNextResult> {
        let result = self.persistent.decompose_next_all(
            &self.fst.borrow(py).inner, &target, &self.ip_univ, &output_symbols,
        );

        let mut per_symbol = rustc_hash::FxHashMap::default();
        let mut symbols: Vec<u32> = result.keys().copied().collect();
        symbols.sort_unstable();

        for (sym, (q_fsa, r_fsa)) in result {
            let q = Py::new(py, RustFsa {
                num_states: q_fsa.num_states, start: q_fsa.start, stop: q_fsa.stop,
                src: q_fsa.arc_src, lbl: q_fsa.arc_lbl, dst: q_fsa.arc_dst,
            })?;
            let r = Py::new(py, RustFsa {
                num_states: r_fsa.num_states, start: r_fsa.start, stop: r_fsa.stop,
                src: r_fsa.arc_src, lbl: r_fsa.arc_lbl, dst: r_fsa.arc_dst,
            })?;
            per_symbol.insert(sym, (q, r));
        }

        Ok(DirtyNextResult { per_symbol, symbols })
    }
}

/// Python-visible result from dirty-state decompose_next().
/// Contains per-symbol (quotient, remainder) pairs.
#[pyclass(unsendable)]
pub struct DirtyNextResult {
    per_symbol: rustc_hash::FxHashMap<u32, (Py<RustFsa>, Py<RustFsa>)>,
    symbols: Vec<u32>,
}

#[pymethods]
impl DirtyNextResult {
    /// Get (quotient, remainder) for a specific output symbol.
    fn get(&self, py: Python<'_>, symbol: u32) -> PyResult<Option<(Py<RustFsa>, Py<RustFsa>)>> {
        Ok(self.per_symbol.get(&symbol).map(|(q, r)| (q.clone_ref(py), r.clone_ref(py))))
    }

    /// Get the quotient FSA for a specific output symbol.
    fn quotient(&self, py: Python<'_>, symbol: u32) -> PyResult<Option<Py<RustFsa>>> {
        Ok(self.per_symbol.get(&symbol).map(|(q, _)| q.clone_ref(py)))
    }

    /// Get the remainder FSA for a specific output symbol.
    fn remainder(&self, py: Python<'_>, symbol: u32) -> PyResult<Option<Py<RustFsa>>> {
        Ok(self.per_symbol.get(&symbol).map(|(_, r)| r.clone_ref(py)))
    }

    /// Get all output symbols.
    fn symbols(&self) -> Vec<u32> {
        self.symbols.clone()
    }
}

// ---------------------------------------------------------------------------
// Peekaboo decomposition
// ---------------------------------------------------------------------------

/// Python-visible peekaboo profiling stats.
#[pyclass]
pub struct RustPeekabooStats {
    #[pyo3(get)]
    pub total_ms: f64,
    #[pyo3(get)]
    pub init_ms: f64,
    #[pyo3(get)]
    pub bfs_ms: f64,
    #[pyo3(get)]
    pub extract_ms: f64,
    #[pyo3(get)]
    pub num_steps: u32,
    #[pyo3(get)]
    pub per_step_visited: Vec<u32>,
    #[pyo3(get)]
    pub per_step_frontier_size: Vec<u32>,
    #[pyo3(get)]
    pub total_bfs_visited: u64,
    #[pyo3(get)]
    pub compute_arcs_ms: f64,
    #[pyo3(get)]
    pub compute_arcs_calls: u64,
    #[pyo3(get)]
    pub intern_ms: f64,
    #[pyo3(get)]
    pub intern_calls: u64,
    #[pyo3(get)]
    pub universal_ms: f64,
    #[pyo3(get)]
    pub universal_calls: u64,
    #[pyo3(get)]
    pub universal_true: u64,
    #[pyo3(get)]
    pub universal_false: u64,
    #[pyo3(get)]
    pub arena_size: u32,
    #[pyo3(get)]
    pub max_powerset_size: usize,
    #[pyo3(get)]
    pub avg_powerset_size: f64,
    #[pyo3(get)]
    pub merged_incoming_states: u32,
    #[pyo3(get)]
    pub merged_incoming_arcs: u64,
    #[pyo3(get)]
    pub eps_cache_clears: u32,
}

#[pymethods]
impl RustPeekabooStats {
    fn __repr__(&self) -> String {
        format!(
            "PeekabooStats(\n\
             \x20 total={:.2}ms  init={:.2}ms  bfs={:.2}ms  extract={:.2}ms\n\
             \x20 steps={}, total_bfs_visited={}\n\
             \x20 compute_arcs: {:.2}ms ({} calls)\n\
             \x20 intern: {:.2}ms ({} calls)\n\
             \x20 universal: {:.2}ms ({} calls, {} true, {} false)\n\
             \x20 arena: {} states, max_pset={}, avg_pset={:.1}\n\
             \x20 incoming: {} states, {} arcs\n\
             )",
            self.total_ms, self.init_ms, self.bfs_ms, self.extract_ms,
            self.num_steps, self.total_bfs_visited,
            self.compute_arcs_ms, self.compute_arcs_calls,
            self.intern_ms, self.intern_calls,
            self.universal_ms, self.universal_calls, self.universal_true, self.universal_false,
            self.arena_size, self.max_powerset_size, self.avg_powerset_size,
            self.merged_incoming_states, self.merged_incoming_arcs,
        )
    }
}

/// Python-visible peekaboo decomposition result.
/// Contains per-symbol (quotient, remainder) pairs.
#[pyclass]
pub struct PeekabooDecompResult {
    /// Map from u32 output symbol to (RustFsa, RustFsa).
    per_symbol: rustc_hash::FxHashMap<u32, (Py<RustFsa>, Py<RustFsa>)>,
    /// All output symbols in sorted order (for iteration).
    symbols: Vec<u32>,
    /// Profiling stats.
    stats: Py<RustPeekabooStats>,
}

#[pymethods]
impl PeekabooDecompResult {
    /// Get (quotient, remainder) for a specific output symbol.
    fn get(&self, py: Python<'_>, symbol: u32) -> PyResult<Option<(Py<RustFsa>, Py<RustFsa>)>> {
        Ok(self.per_symbol.get(&symbol).map(|(q, r)| (q.clone_ref(py), r.clone_ref(py))))
    }

    /// Get the quotient FSA for a specific output symbol.
    fn quotient(&self, py: Python<'_>, symbol: u32) -> PyResult<Option<Py<RustFsa>>> {
        Ok(self.per_symbol.get(&symbol).map(|(q, _)| q.clone_ref(py)))
    }

    /// Get the remainder FSA for a specific output symbol.
    fn remainder(&self, py: Python<'_>, symbol: u32) -> PyResult<Option<Py<RustFsa>>> {
        Ok(self.per_symbol.get(&symbol).map(|(_, r)| r.clone_ref(py)))
    }

    /// Get all output symbols.
    fn symbols(&self) -> Vec<u32> {
        self.symbols.clone()
    }

    /// Get profiling stats.
    fn stats(&self, py: Python<'_>) -> Py<RustPeekabooStats> {
        self.stats.clone_ref(py)
    }
}

// ---------------------------------------------------------------------------
// Dirty (incremental) peekaboo decomposition
// ---------------------------------------------------------------------------

/// Python-visible incremental peekaboo decomposition wrapper.
/// Persists peekaboo BFS state across calls; on prefix extension, only
/// runs the new step(s) instead of rebuilding from scratch.
#[pyclass(unsendable)]
pub struct RustDirtyPeekabooDecomp {
    fst: Py<RustFst>,
    persistent: peekaboo::DirtyPeekaboo,
}

#[pymethods]
impl RustDirtyPeekabooDecomp {
    #[new]
    fn new(py: Python<'_>, fst: Py<RustFst>) -> Self {
        let persistent = peekaboo::DirtyPeekaboo::new(&fst.borrow(py).inner);
        RustDirtyPeekabooDecomp { fst, persistent }
    }

    /// Decompose the FST for the given target, returning a lightweight
    /// PeekabooBeamView for beam search (no FSA construction).
    fn decompose_for_beam(&mut self, py: Python<'_>, target: Vec<u32>) -> PyResult<PeekabooBeamView> {
        let fst = &self.fst.borrow(py).inner;
        // Run BFS (reuses dirty-state incremental logic)
        self.persistent.decompose_bfs_only(fst, &target);

        let step_n = target.len() as u16;
        let start_id = self.persistent.global_start_id();
        let preimage_stops = self.persistent.compute_preimage_stops(fst, step_n);
        let resume_frontiers_raw = self.persistent.compute_resume_frontiers();

        let idx_to_sym = self.persistent.idx_to_sym();
        let reachable_flags = self.persistent.reachable_flags();

        // Convert decomp_q: u16 idx -> u32 sym, filter to reachable
        let mut decomp_q: Vec<(u32, Vec<u32>)> = Vec::new();
        for (&y_idx, sids) in self.persistent.decomp_q() {
            let sym = idx_to_sym[y_idx as usize];
            let filtered: Vec<u32> = sids.iter()
                .filter(|&&sid| (sid as usize) < reachable_flags.len() && reachable_flags[sid as usize])
                .copied().collect();
            if !filtered.is_empty() {
                decomp_q.push((sym, filtered));
            }
        }

        // Convert decomp_r: u16 idx -> u32 sym, filter to reachable
        let mut decomp_r: Vec<(u32, Vec<u32>)> = Vec::new();
        for (&y_idx, sids) in self.persistent.decomp_r() {
            let sym = idx_to_sym[y_idx as usize];
            let filtered: Vec<u32> = sids.iter()
                .filter(|&&sid| (sid as usize) < reachable_flags.len() && reachable_flags[sid as usize])
                .copied().collect();
            if !filtered.is_empty() {
                decomp_r.push((sym, filtered));
            }
        }

        // Convert resume_frontiers: u16 idx -> u32 sym
        let mut resume_frontiers: Vec<(u32, Vec<u32>)> = Vec::new();
        for (y_idx, sids) in resume_frontiers_raw {
            let sym = idx_to_sym[y_idx as usize];
            resume_frontiers.push((sym, sids));
        }

        Ok(PeekabooBeamView {
            start_id,
            preimage_stops,
            decomp_q,
            decomp_r,
            resume_frontiers,
        })
    }

    /// Return the arcs from a DFA state: Vec<(input_label_u32, dest_sid)>.
    /// Called per-particle during beam search.
    fn arcs_for(&self, py: Python<'_>, state_id: u32) -> Vec<(u32, u32)> {
        let _ = py;
        self.persistent.arcs_from(state_id).to_vec()
    }

    /// Decompose the FST for the given target, returning per-symbol Q/R.
    #[pyo3(signature = (target, minimize=false))]
    fn decompose(&mut self, py: Python<'_>, target: Vec<u32>, minimize: bool) -> PyResult<PeekabooDecompResult> {
        let result = self.persistent.decompose(&self.fst.borrow(py).inner, &target);

        let mut per_symbol = rustc_hash::FxHashMap::default();
        let mut symbols: Vec<u32> = result.per_symbol.keys().copied().collect();
        symbols.sort_unstable();

        for (sym, (q_fsa, r_fsa)) in result.per_symbol {
            let (q_fsa, r_fsa) = if minimize {
                (minimize::minimize(&q_fsa), minimize::minimize(&r_fsa))
            } else {
                (q_fsa, r_fsa)
            };

            let q = Py::new(py, RustFsa {
                num_states: q_fsa.num_states, start: q_fsa.start, stop: q_fsa.stop,
                src: q_fsa.arc_src, lbl: q_fsa.arc_lbl, dst: q_fsa.arc_dst,
            })?;
            let r = Py::new(py, RustFsa {
                num_states: r_fsa.num_states, start: r_fsa.start, stop: r_fsa.stop,
                src: r_fsa.arc_src, lbl: r_fsa.arc_lbl, dst: r_fsa.arc_dst,
            })?;

            per_symbol.insert(sym, (q, r));
        }

        let s = &result.stats;
        let stats = Py::new(py, RustPeekabooStats {
            total_ms: s.total_ms,
            init_ms: s.init_ms,
            bfs_ms: s.bfs_ms,
            extract_ms: s.extract_ms,
            num_steps: s.num_steps,
            per_step_visited: s.per_step_visited.clone(),
            per_step_frontier_size: s.per_step_frontier_size.clone(),
            total_bfs_visited: s.total_bfs_visited,
            compute_arcs_ms: s.compute_arcs_ms,
            compute_arcs_calls: s.compute_arcs_calls,
            intern_ms: s.intern_ms,
            intern_calls: s.intern_calls,
            universal_ms: s.universal_ms,
            universal_calls: s.universal_calls,
            universal_true: s.universal_true,
            universal_false: s.universal_false,
            arena_size: s.arena_size,
            max_powerset_size: s.max_powerset_size,
            avg_powerset_size: s.avg_powerset_size,
            merged_incoming_states: s.merged_incoming_states,
            merged_incoming_arcs: s.merged_incoming_arcs,
            eps_cache_clears: s.eps_cache_clears,
        })?;

        Ok(PeekabooDecompResult { per_symbol, symbols, stats })
    }
}

// ---------------------------------------------------------------------------
// PeekabooBeamView: lightweight snapshot for beam search
// ---------------------------------------------------------------------------

/// Lightweight snapshot of DirtyPeekaboo state for beam search.
/// Returned by decompose_for_beam(); avoids FSA construction.
#[pyclass]
pub struct PeekabooBeamView {
    #[pyo3(get)]
    start_id: u32,
    #[pyo3(get)]
    preimage_stops: Vec<u32>,
    decomp_q: Vec<(u32, Vec<u32>)>,
    decomp_r: Vec<(u32, Vec<u32>)>,
    resume_frontiers: Vec<(u32, Vec<u32>)>,
}

#[pymethods]
impl PeekabooBeamView {
    /// Per-symbol quotient stop states: Vec<(output_sym_u32, stop_sids)>.
    fn decomp_q(&self) -> Vec<(u32, Vec<u32>)> {
        self.decomp_q.clone()
    }

    /// Per-symbol remainder stop states: Vec<(output_sym_u32, stop_sids)>.
    fn decomp_r(&self) -> Vec<(u32, Vec<u32>)> {
        self.decomp_r.clone()
    }

    /// Per-symbol resume frontier states: Vec<(output_sym_u32, frontier_sids)>.
    fn resume_frontiers(&self) -> Vec<(u32, Vec<u32>)> {
        self.resume_frontiers.clone()
    }
}

