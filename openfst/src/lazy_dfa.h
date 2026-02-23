#pragma once
/// LazyPeekabooDFA: lazy DFA for FusedTransducedLM integration.
/// Same API as RustLazyPeekabooDFA: new_step, start_ids, arcs, run, classify.

#include "fst_convert.h"
#include "peekaboo_fst.h"
#include "peekaboo_universality.h"
#include "powerset.h"
#include "universality.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace transduction {

struct ExpandEntry {
    uint32_t sid;
    ClassifyResult cr;
    std::vector<std::pair<uint32_t, uint32_t>> arcs;  // empty if quotient
};

class LazyPeekabooDFA {
public:
    explicit LazyPeekabooDFA(const FstData* fst);

    /// Set up for a new target step.
    void new_step(const std::vector<uint32_t>& target);

    /// Return start state IDs.
    std::vector<uint32_t> start_ids() const;

    /// Return arcs from a DFA state: (label, dest_id).
    std::vector<std::pair<uint32_t, uint32_t>> arcs(uint32_t sid);

    /// Run a source path from start states.
    std::optional<uint32_t> run(const std::vector<uint32_t>& source_path);

    /// Classify a DFA state.
    ClassifyResult classify(uint32_t sid);

    /// Batch classify + arcs for a set of DFA states (reduces Python↔C++ round-trips).
    std::vector<ExpandEntry> expand_batch(const std::vector<uint32_t>& sids);

    /// Decode a DFA state to NFA elements.
    std::vector<std::tuple<uint32_t, uint16_t, uint16_t, bool>> decode_state(uint32_t sid) const;

    /// Map from sym_idx -> original symbol u32.
    std::vector<uint32_t> idx_to_sym_map() const;

private:
    const FstData* fst_;
    std::vector<uint32_t> current_target_;

    // Symbol index mapping
    std::unordered_map<uint32_t, uint16_t> sym_to_idx_;
    std::vector<uint32_t> idx_to_sym_;

    // Per-step state
    PowersetArena arena_;
    EpsCache eps_cache_;
    std::vector<uint32_t> start_ids_;
    std::optional<PeekabooNFA> nfa_;

    // Arc cache: dfa_state -> [(label, dest)]
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> arc_cache_;

    // Classify cache: dfa_state -> ClassifyResult (cleared each step)
    std::unordered_map<uint32_t, ClassifyResult> classify_cache_;

    // Reusable buffer for compute_all_arcs_into (avoids map reallocation)
    std::unordered_map<uint32_t, std::vector<uint64_t>> arcs_buf_;

    // Per-symbol universality filters (keyed by y_idx), reset each step
    std::unordered_map<uint16_t, PerSymbolUnivFilter> univ_filters_;

    void ensure_arcs(uint32_t sid);
};

}  // namespace transduction
