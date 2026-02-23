#pragma once
/// PrecoverNFA: mirrors Rust precover.rs
/// States are (fst_state, buf_pos) packed as u64.

#include "fst_convert.h"
#include "powerset.h"

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace transduction {

/// Pack (fst_state, buf_pos) into u64 using stride.
inline uint64_t pack_precover(uint32_t fst_state, uint32_t buf_pos, uint64_t stride) {
    return static_cast<uint64_t>(fst_state) * stride + buf_pos;
}

/// Unpack u64 back to (fst_state, buf_pos).
inline std::pair<uint32_t, uint32_t> unpack_precover(uint64_t packed, uint64_t stride) {
    return {static_cast<uint32_t>(packed / stride),
            static_cast<uint32_t>(packed % stride)};
}

class PrecoverNFA {
public:
    const FstData* fst;
    const uint32_t* target;
    uint32_t target_len;
    uint64_t stride;

    // Epsilon closure cache: packed_state -> (closure, max_buf_pos)
    std::unordered_map<uint64_t, std::pair<std::vector<uint64_t>, uint32_t>> eps_cache;
    uint64_t eps_cache_hits = 0;
    uint64_t eps_cache_misses = 0;

    PrecoverNFA(const FstData* fst, const uint32_t* target, uint32_t target_len)
        : fst(fst), target(target), target_len(target_len),
          stride(static_cast<uint64_t>(target_len) + 1) {}

    PrecoverNFA(const FstData* fst, const uint32_t* target, uint32_t target_len, uint64_t stride)
        : fst(fst), target(target), target_len(target_len), stride(stride) {}

    uint64_t pack_state(uint32_t fst_state, uint32_t buf_pos) const {
        return pack_precover(fst_state, buf_pos, stride);
    }

    std::pair<uint32_t, uint32_t> unpack_state(uint64_t packed) const {
        return unpack_precover(packed, stride);
    }

    bool is_final(uint64_t packed) const {
        auto [fst_state, buf_pos] = unpack_state(packed);
        return fst->is_final[fst_state] && buf_pos == target_len;
    }

    /// All arcs from a packed NFA state: (input_symbol, destination).
    std::vector<std::pair<uint32_t, uint64_t>> arcs(uint64_t packed) const;

    /// Epsilon-only successors (for eps_closure).
    std::vector<uint64_t> arcs_eps(uint64_t packed) const;

    /// Start states.
    std::vector<uint64_t> start_states() const;

    /// Is an NFA state "productive"?
    bool is_productive(uint64_t packed) const;

    /// Epsilon closure of a single state, cached.
    const std::vector<uint64_t>& eps_closure_single(uint64_t state);

    /// Epsilon closure of a set of states.
    void eps_closure_set(const std::vector<uint64_t>& states, std::vector<uint64_t>& out);

    /// Batch-compute all non-epsilon arcs from an epsilon-closed powerset state.
    /// Returns vector of (symbol, sorted+deduped destination set).
    std::vector<std::pair<uint32_t, std::vector<uint64_t>>> compute_all_arcs(
        const std::vector<uint64_t>& states);
};

}  // namespace transduction
