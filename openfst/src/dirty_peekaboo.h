#pragma once
/// DirtyPeekaboo: dirty-state incremental peekaboo decomposition.
/// Mirrors Rust peekaboo.rs DirtyPeekaboo struct.

#include "fst_convert.h"
#include "decompose.h"
#include "peekaboo_fst.h"
#include "powerset.h"
#include "universality.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace transduction {

/// Per-symbol decomposition result.
struct PeekabooSymResult {
    FsaResult quotient;
    FsaResult remainder;
};

/// Result of a peekaboo decompose call.
struct PeekabooDecompResult {
    std::unordered_map<uint32_t, PeekabooSymResult> by_symbol;
    std::vector<uint32_t> symbols;
};

class DirtyPeekaboo {
public:
    explicit DirtyPeekaboo(const FstData* fst);

    /// Decompose: compute per-symbol Q/R for target prefix.
    PeekabooDecompResult decompose(const std::vector<uint32_t>& target, bool minimize = false);

private:
    const FstData* fst_;
    std::unordered_map<uint32_t, uint16_t> sym_to_idx_;
    std::vector<uint32_t> idx_to_sym_;

    // Persistent state across calls
    PowersetArena arena_;
    EpsCache eps_cache_;

    uint32_t prev_target_len_ = 0;

    void build_per_symbol_fsa(
        uint32_t sid,
        const PowersetArena& arena,
        const PeekabooNFA& nfa,
        uint16_t step_n,
        std::unordered_map<uint32_t, std::vector<uint32_t>>& q_stops,
        std::unordered_map<uint32_t, std::vector<uint32_t>>& r_stops
    );
};

}  // namespace transduction
