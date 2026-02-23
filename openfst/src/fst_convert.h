#pragma once
/// Convert parallel arrays (same format as RustFst constructor) to OpenFST VectorFst.

#include <cstdint>
#include <memory>
#include <vector>

#include <fst/fstlib.h>

namespace transduction {

/// Our epsilon convention in the integer representation (u32::MAX).
/// OpenFST uses 0 for epsilon.  Conversion happens in make_vector_fst.
constexpr uint32_t INTERNAL_EPSILON = 0xFFFFFFFF;

/// CSR-style FST representation mirroring Rust's Fst struct.
/// Provides output-group directory and epsilon-input side table for fast access.
struct FstData {
    uint32_t num_states;
    std::vector<uint32_t> start_states;
    std::vector<bool> is_final;

    // CSR arc storage (sorted by (src, output, input))
    struct Arc {
        uint32_t input;
        uint32_t output;
        uint32_t dest;
    };
    std::vector<uint32_t> offsets;  // length num_states+1
    std::vector<Arc> arcs;

    // Output-group directory
    struct OutputGroup {
        uint32_t label;
        uint32_t end;  // absolute index into arcs[]
    };
    std::vector<uint32_t> group_offsets;  // length num_states+1
    std::vector<OutputGroup> output_groups;

    // Epsilon-input side table
    struct EpsArc {
        uint32_t output;
        uint32_t dest;
    };
    std::vector<uint32_t> eps_offsets;  // length num_states+1
    std::vector<EpsArc> eps_arcs;

    std::vector<uint32_t> source_alphabet;
    std::vector<bool> has_non_eps_input;

    // IP-universal states (greatest fixpoint)
    std::vector<bool> ip_universal;

    // --- Accessors (match Rust Fst methods) ---

    const Arc* arcs_from(uint32_t state, size_t& count) const {
        size_t lo = offsets[state];
        size_t hi = offsets[state + 1];
        count = hi - lo;
        return arcs.data() + lo;
    }

    // Binary search on output-group directory for arcs with given output label
    const Arc* arcs_by_output(uint32_t state, uint32_t output, size_t& count) const {
        size_t glo = group_offsets[state];
        size_t ghi = group_offsets[state + 1];
        count = 0;
        if (glo == ghi) return nullptr;

        // Binary search
        size_t lo = glo, hi = ghi;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (output_groups[mid].label < output) lo = mid + 1;
            else hi = mid;
        }
        if (lo >= ghi || output_groups[lo].label != output) return nullptr;

        uint32_t arc_start = (lo == glo) ? offsets[state] : output_groups[lo - 1].end;
        uint32_t arc_end = output_groups[lo].end;
        count = arc_end - arc_start;
        return arcs.data() + arc_start;
    }

    const EpsArc* eps_input_arcs(uint32_t state, size_t& count) const {
        size_t lo = eps_offsets[state];
        size_t hi = eps_offsets[state + 1];
        count = hi - lo;
        return eps_arcs.data() + lo;
    }

    // Binary search within eps_arcs for a specific output label
    const EpsArc* eps_input_arcs_by_output(uint32_t state, uint32_t output, size_t& count) const {
        size_t elo = eps_offsets[state];
        size_t ehi = eps_offsets[state + 1];
        count = 0;
        if (elo == ehi) return nullptr;

        // Find first with output >= target
        size_t lo = elo, hi = ehi;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (eps_arcs[mid].output < output) lo = mid + 1;
            else hi = mid;
        }
        size_t start = lo;
        // Find first with output > target
        hi = ehi;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (eps_arcs[mid].output <= output) lo = mid + 1;
            else hi = mid;
        }
        count = lo - start;
        return (count > 0) ? (eps_arcs.data() + start) : nullptr;
    }
};

/// Build FstData from parallel arrays (same format as RustFst::new).
/// Labels use INTERNAL_EPSILON (u32::MAX) for epsilon.
std::unique_ptr<FstData> make_fst_data(
    uint32_t num_states,
    const std::vector<uint32_t>& start_states,
    const std::vector<uint32_t>& final_states,
    const std::vector<uint32_t>& arc_src,
    const std::vector<uint32_t>& arc_in,
    const std::vector<uint32_t>& arc_out,
    const std::vector<uint32_t>& arc_dst,
    const std::vector<uint32_t>& source_alphabet
);

/// Build an OpenFST VectorFst from the same parallel arrays.
/// Labels are remapped: INTERNAL_EPSILON -> 0 (OpenFST epsilon).
/// Non-epsilon labels are offset by +1 to avoid collision with epsilon=0.
std::unique_ptr<fst::StdVectorFst> make_vector_fst(
    uint32_t num_states,
    const std::vector<uint32_t>& start_states,
    const std::vector<uint32_t>& final_states,
    const std::vector<uint32_t>& arc_src,
    const std::vector<uint32_t>& arc_in,
    const std::vector<uint32_t>& arc_out,
    const std::vector<uint32_t>& arc_dst
);

}  // namespace transduction
