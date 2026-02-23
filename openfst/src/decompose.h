#pragma once
/// One-shot decomposition: fused BFS with determinization + universality + Q/R split.
/// Mirrors Rust decompose.rs::decompose().

#include "fst_convert.h"
#include "universality.h"

#include <cstdint>
#include <vector>

namespace transduction {

/// Result FSA as parallel arrays (easy to pass across Cython boundary).
struct FsaResult {
    uint32_t num_states;
    std::vector<uint32_t> start;
    std::vector<uint32_t> stop;
    std::vector<uint32_t> arc_src;
    std::vector<uint32_t> arc_lbl;
    std::vector<uint32_t> arc_dst;
};

/// Decomposition result: Q, R, and profiling stats.
struct DecompResult {
    FsaResult quotient;
    FsaResult remainder;
    ProfileStats stats;
};

/// One-shot decomposition using our CSR FST data.
DecompResult decompose(const FstData& fst, const uint32_t* target, uint32_t target_len);

}  // namespace transduction
