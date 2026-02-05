use pyo3::prelude::*;
use crate::fst::Fst;
use crate::decompose;
use crate::token_decompose;
use crate::peekaboo;

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

/// Perform the fused decomposition: given a Rust FST and a target sequence,
/// compute the quotient (Q) and remainder (R) automata.
#[pyfunction]
#[pyo3(signature = (fst, target, force_generic=false))]
pub fn rust_decompose(py: Python<'_>, fst: &RustFst, target: Vec<u32>, force_generic: bool) -> PyResult<DecompResult> {
    let result = if fst.inner.all_input_universal && !force_generic {
        token_decompose::decompose_token_level(&fst.inner, &target)
    } else {
        decompose::decompose(&fst.inner, &target)
    };

    let q = Py::new(
        py,
        RustFsa {
            num_states: result.quotient.num_states,
            start: result.quotient.start,
            stop: result.quotient.stop,
            src: result.quotient.arc_src,
            lbl: result.quotient.arc_lbl,
            dst: result.quotient.arc_dst,
        },
    )?;

    let r = Py::new(
        py,
        RustFsa {
            num_states: result.remainder.num_states,
            start: result.remainder.start,
            stop: result.remainder.stop,
            src: result.remainder.arc_src,
            lbl: result.remainder.arc_lbl,
            dst: result.remainder.arc_dst,
        },
    )?;

    let s = &result.stats;
    let stats = Py::new(
        py,
        RustProfileStats {
            total_ms: s.total_ms,
            init_ms: s.init_ms,
            bfs_ms: s.bfs_ms,
            compute_arcs_ms: s.compute_arcs_ms,
            compute_arcs_calls: s.compute_arcs_calls,
            intern_ms: s.intern_ms,
            intern_calls: s.intern_calls,
            universal_ms: s.universal_ms,
            universal_calls: s.universal_calls,
            universal_true: s.universal_true,
            universal_false: s.universal_false,
            universal_sub_bfs_states: s.universal_sub_bfs_states,
            universal_compute_arcs_calls: s.universal_compute_arcs_calls,
            dfa_states: s.dfa_states,
            total_arcs: s.total_arcs,
            q_stops: s.q_stops,
            r_stops: s.r_stops,
            max_powerset_size: s.max_powerset_size,
            avg_powerset_size: s.avg_powerset_size,
            eps_cache_hits: s.eps_cache_hits,
            eps_cache_misses: s.eps_cache_misses,
        },
    )?;

    Ok(DecompResult {
        quotient: q,
        remainder: r,
        stats,
    })
}

// ---------------------------------------------------------------------------
// Peekaboo decomposition
// ---------------------------------------------------------------------------

/// Python-visible peekaboo decomposition result.
/// Contains per-symbol (quotient, remainder) pairs.
#[pyclass]
pub struct PeekabooDecompResult {
    /// Map from u32 output symbol to (RustFsa, RustFsa).
    per_symbol: rustc_hash::FxHashMap<u32, (Py<RustFsa>, Py<RustFsa>)>,
    /// All output symbols in sorted order (for iteration).
    symbols: Vec<u32>,
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
}

/// Perform peekaboo decomposition: given a Rust FST and a target sequence,
/// compute per-symbol quotient and remainder automata for the next target symbol.
#[pyfunction]
pub fn rust_peekaboo(py: Python<'_>, fst: &RustFst, target: Vec<u32>) -> PyResult<PeekabooDecompResult> {
    let result = peekaboo::peekaboo_decompose(&fst.inner, &target);

    let mut per_symbol = rustc_hash::FxHashMap::default();
    let mut symbols: Vec<u32> = result.per_symbol.keys().copied().collect();
    symbols.sort_unstable();

    for (sym, (q_fsa, r_fsa)) in result.per_symbol {
        let q = Py::new(
            py,
            RustFsa {
                num_states: q_fsa.num_states,
                start: q_fsa.start,
                stop: q_fsa.stop,
                src: q_fsa.arc_src,
                lbl: q_fsa.arc_lbl,
                dst: q_fsa.arc_dst,
            },
        )?;

        let r = Py::new(
            py,
            RustFsa {
                num_states: r_fsa.num_states,
                start: r_fsa.start,
                stop: r_fsa.stop,
                src: r_fsa.arc_src,
                lbl: r_fsa.arc_lbl,
                dst: r_fsa.arc_dst,
            },
        )?;

        per_symbol.insert(sym, (q, r));
    }

    Ok(PeekabooDecompResult { per_symbol, symbols })
}
