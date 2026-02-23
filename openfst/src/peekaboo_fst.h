#pragma once
/// PeekabooNFA: mirrors Rust peekaboo.rs PeekabooNFAMapped.
///
/// NFA state packing:
///   bits [63:32] = fst_state (u32)
///   bits [31:17] = buf_len   (u15, max 32767)
///   bits [16:1]  = extra_sym_idx (u16, 0xFFFF = NO_EXTRA for on-target)
///   bit  [0]     = truncated

#include "fst_convert.h"
#include "powerset.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace transduction {

constexpr uint16_t NO_EXTRA = 0xFFFF;

/// Epsilon-closure cache: packed NFA state -> (shared closure vec, max buf_len).
using EpsCache = std::unordered_map<uint64_t,
    std::pair<std::shared_ptr<const std::vector<uint64_t>>, uint16_t>>;

inline uint64_t pack_peekaboo(uint32_t fst_state, uint16_t buf_len, uint16_t extra_sym, bool truncated) {
    return (static_cast<uint64_t>(fst_state) << 32)
         | (static_cast<uint64_t>(buf_len) << 17)
         | (static_cast<uint64_t>(extra_sym) << 1)
         | static_cast<uint64_t>(truncated);
}

inline void unpack_peekaboo(uint64_t packed, uint32_t& fst_state, uint16_t& buf_len,
                            uint16_t& extra_sym, bool& truncated) {
    fst_state = static_cast<uint32_t>(packed >> 32);
    buf_len = static_cast<uint16_t>((packed >> 17) & 0x7FFF);
    extra_sym = static_cast<uint16_t>((packed >> 1) & 0xFFFF);
    truncated = (packed & 1) != 0;
}

/// Classify result for a DFA state.
struct ClassifyResult {
    int32_t quotient_sym;    // -1 if none
    std::vector<uint32_t> remainder_syms;
    bool is_preimage;
    bool has_truncated;
    std::vector<uint32_t> trunc_output_syms;
};

class PeekabooNFA {
public:
    const FstData* fst;
    const uint32_t* full_target;
    uint32_t full_target_len;
    uint16_t step_n;
    const std::unordered_map<uint32_t, uint16_t>* sym_to_idx;

    PeekabooNFA(const FstData* fst, const uint32_t* full_target, uint32_t full_target_len,
                uint16_t step_n, const std::unordered_map<uint32_t, uint16_t>* sym_to_idx)
        : fst(fst), full_target(full_target), full_target_len(full_target_len),
          step_n(step_n), sym_to_idx(sym_to_idx) {}

    std::vector<uint64_t> start_states() const;
    bool is_final(uint64_t packed) const;
    bool is_productive(uint64_t packed) const;

    struct EffState {
        uint16_t eff_n;
        uint16_t eff_extra;
        bool is_valid;
    };
    EffState effective_state(uint16_t buf_len, uint16_t extra_sym, bool truncated) const;

    std::vector<std::pair<uint32_t, uint64_t>> arcs(uint64_t packed) const;
    std::shared_ptr<const std::vector<uint64_t>> eps_closure_single(
        uint64_t state, EpsCache& cache) const;
    void eps_closure_set(const std::vector<uint64_t>& states, std::vector<uint64_t>& out,
        EpsCache& cache) const;

    std::vector<std::pair<uint32_t, std::vector<uint64_t>>> compute_all_arcs(
        const std::vector<uint64_t>& states, EpsCache& cache) const;

    /// Buffered variant: reuses by_symbol map across calls to avoid reallocation.
    void compute_all_arcs_into(
        const std::vector<uint64_t>& states, EpsCache& cache,
        std::unordered_map<uint32_t, std::vector<uint64_t>>& buf) const;
};

}  // namespace transduction
