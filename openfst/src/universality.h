#pragma once
/// UniversalityFilter: mirrors Rust decompose.rs UniversalityFilter.
/// Greatest-fixpoint universality detection with witness/superset/subset caches.

#include "fst_convert.h"
#include "powerset.h"
#include "precover_fst.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace transduction {

struct ProfileStats {
    double total_ms = 0;
    double init_ms = 0;
    double bfs_ms = 0;
    double compute_arcs_ms = 0;
    uint64_t compute_arcs_calls = 0;
    double intern_ms = 0;
    uint64_t intern_calls = 0;
    double universal_ms = 0;
    uint64_t universal_calls = 0;
    uint64_t universal_true = 0;
    uint64_t universal_false = 0;
    uint64_t universal_sub_bfs_states = 0;
    uint64_t universal_compute_arcs_calls = 0;
    uint32_t dfa_states = 0;
    uint64_t total_arcs = 0;
    uint32_t q_stops = 0;
    uint32_t r_stops = 0;
    size_t max_powerset_size = 0;
    double avg_powerset_size = 0;
    uint64_t eps_cache_hits = 0;
    uint64_t eps_cache_misses = 0;
};

class UniversalityFilter {
public:
    std::unordered_set<uint64_t> witnesses;

    // Positive cache: element-indexed
    std::unordered_map<uint64_t, std::vector<uint32_t>> pos_index;
    std::vector<size_t> pos_sizes;

    // Negative cache: element-indexed
    std::unordered_map<uint64_t, std::vector<uint32_t>> neg_index;
    uint32_t neg_next = 0;

    UniversalityFilter() = default;

    /// Build with pre-computed ip_universal states.
    UniversalityFilter(const std::vector<bool>& ip_univ, uint32_t target_len, uint64_t stride);

    void add_pos(const std::vector<uint64_t>& nfa_set);
    void add_neg(const std::vector<uint64_t>& nfa_set);

    bool has_pos_subset(const std::vector<uint64_t>& nfa_set) const;
    bool has_neg_superset(const std::vector<uint64_t>& nfa_set) const;

    bool bfs_universal(
        uint32_t sid,
        PrecoverNFA& nfa,
        PowersetArena& arena,
        size_t num_source_symbols,
        ProfileStats& stats
    );

    bool is_universal(
        uint32_t sid,
        PrecoverNFA& nfa,
        PowersetArena& arena,
        size_t num_source_symbols,
        ProfileStats& stats
    );
};

}  // namespace transduction
