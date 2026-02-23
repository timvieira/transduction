# cython: language_level=3
# C++ declarations for _openfst_decomp

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.memory cimport unique_ptr
from libcpp cimport bool as cbool
from libc.stdint cimport uint16_t, uint32_t, uint64_t, int32_t
from libcpp.optional cimport optional

cdef extern from "fst_convert.h" namespace "transduction":
    cdef cppclass FstData:
        uint32_t num_states

    unique_ptr[FstData] make_fst_data(
        uint32_t num_states,
        const vector[uint32_t]& start_states,
        const vector[uint32_t]& final_states,
        const vector[uint32_t]& arc_src,
        const vector[uint32_t]& arc_in,
        const vector[uint32_t]& arc_out,
        const vector[uint32_t]& arc_dst,
        const vector[uint32_t]& source_alphabet
    )

cdef extern from "decompose.h" namespace "transduction":
    cdef cppclass FsaResult:
        uint32_t num_states
        vector[uint32_t] start
        vector[uint32_t] stop
        vector[uint32_t] arc_src
        vector[uint32_t] arc_lbl
        vector[uint32_t] arc_dst

    cdef cppclass ProfileStats:
        double total_ms
        double init_ms
        double bfs_ms
        double compute_arcs_ms
        uint64_t compute_arcs_calls
        double intern_ms
        uint64_t intern_calls
        double universal_ms
        uint64_t universal_calls
        uint64_t universal_true
        uint64_t universal_false
        uint64_t universal_sub_bfs_states
        uint64_t universal_compute_arcs_calls
        uint32_t dfa_states
        uint64_t total_arcs
        uint32_t q_stops
        uint32_t r_stops
        size_t max_powerset_size
        double avg_powerset_size
        uint64_t eps_cache_hits
        uint64_t eps_cache_misses

    cdef cppclass DecompResult:
        FsaResult quotient
        FsaResult remainder
        ProfileStats stats

    DecompResult decompose(const FstData& fst, const uint32_t* target, uint32_t target_len)

cdef extern from "peekaboo_fst.h" namespace "transduction":
    cdef cppclass ClassifyResult:
        int32_t quotient_sym
        vector[uint32_t] remainder_syms
        cbool is_preimage
        cbool has_truncated
        vector[uint32_t] trunc_output_syms

cdef extern from "lazy_dfa.h" namespace "transduction":
    cdef cppclass ExpandEntry:
        uint32_t sid
        ClassifyResult cr
        vector[pair[uint32_t, uint32_t]] arcs

    cdef cppclass LazyPeekabooDFA:
        LazyPeekabooDFA(const FstData* fst) except +
        void new_step(const vector[uint32_t]& target)
        vector[uint32_t] start_ids() const
        vector[pair[uint32_t, uint32_t]] arcs(uint32_t sid)
        optional[uint32_t] run(const vector[uint32_t]& source_path)
        ClassifyResult classify(uint32_t sid)
        vector[ExpandEntry] expand_batch(const vector[uint32_t]& sids)
        vector[uint32_t] idx_to_sym_map() const

cdef extern from "dirty_peekaboo.h" namespace "transduction":
    cdef cppclass PeekabooSymResult:
        FsaResult quotient
        FsaResult remainder

    cdef cppclass PeekabooDecompResult:
        unordered_map[uint32_t, PeekabooSymResult] by_symbol
        vector[uint32_t] symbols

    cdef cppclass DirtyPeekaboo:
        DirtyPeekaboo(const FstData* fst) except +
        PeekabooDecompResult decompose(const vector[uint32_t]& target, cbool minimize)
