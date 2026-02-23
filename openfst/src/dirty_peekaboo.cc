#include "dirty_peekaboo.h"
#include "peekaboo_universality.h"

#include <algorithm>
#include <deque>
#include <unordered_set>

namespace transduction {

DirtyPeekaboo::DirtyPeekaboo(const FstData* fst) : fst_(fst) {
    // Build symbol index from all non-eps labels
    std::unordered_set<uint32_t> all_syms;
    for (const auto& arc : fst->arcs) {
        if (arc.input != INTERNAL_EPSILON) all_syms.insert(arc.input);
        if (arc.output != INTERNAL_EPSILON) all_syms.insert(arc.output);
    }
    uint16_t idx = 0;
    for (uint32_t sym : all_syms) {
        sym_to_idx_[sym] = idx;
        if (idx >= idx_to_sym_.size()) idx_to_sym_.resize(idx + 1);
        idx_to_sym_[idx] = sym;
        idx++;
    }
}

void DirtyPeekaboo::build_per_symbol_fsa(
    uint32_t sid,
    const PowersetArena& arena,
    const PeekabooNFA& nfa,
    uint16_t step_n,
    std::unordered_map<uint32_t, std::vector<uint32_t>>& q_stops,
    std::unordered_map<uint32_t, std::vector<uint32_t>>& r_stops
) {
    // Now handled in decompose() via per-symbol universality
}

/// Build a trimmed FSA from stop states via backward BFS.
/// Arcs from Q-stop states are excluded (they are sinks).
static FsaResult collect_arcs_trimmed(
    uint32_t start_id,
    const std::vector<uint32_t>& stops,
    uint32_t num_states,
    const std::vector<uint32_t>& all_arc_src,
    const std::vector<uint32_t>& all_arc_lbl,
    const std::vector<uint32_t>& all_arc_dst,
    const std::unordered_set<uint32_t>& q_stop_set,  // global Q-stop states
    const std::vector<bool>& reachable
) {
    if (stops.empty()) {
        return {0, {}, {}, {}, {}, {}};
    }

    // Build reverse arcs
    std::unordered_map<uint32_t, std::vector<uint32_t>> reverse;  // dst -> [src]
    for (size_t i = 0; i < all_arc_src.size(); i++) {
        reverse[all_arc_dst[i]].push_back(all_arc_src[i]);
    }

    // Backward BFS from stops
    std::vector<bool> backward(num_states, false);
    std::deque<uint32_t> bfs;
    for (uint32_t s : stops) {
        if (s < num_states) {
            backward[s] = true;
            bfs.push_back(s);
        }
    }
    while (!bfs.empty()) {
        uint32_t sid = bfs.front();
        bfs.pop_front();
        auto it = reverse.find(sid);
        if (it != reverse.end()) {
            for (uint32_t src : it->second) {
                if (src < num_states && !backward[src] && reachable[src]) {
                    backward[src] = true;
                    bfs.push_back(src);
                }
            }
        }
    }

    std::vector<uint32_t> start;
    if (start_id < num_states && backward[start_id]) {
        start.push_back(start_id);
    }

    // Collect arcs: skip arcs FROM Q-stop states (they are sinks)
    std::vector<uint32_t> arc_src, arc_lbl, arc_dst;
    for (size_t i = 0; i < all_arc_src.size(); i++) {
        uint32_t s = all_arc_src[i];
        uint32_t d = all_arc_dst[i];
        if (backward[s] && backward[d] && q_stop_set.find(s) == q_stop_set.end()) {
            arc_src.push_back(s);
            arc_lbl.push_back(all_arc_lbl[i]);
            arc_dst.push_back(d);
        }
    }

    return {num_states, std::move(start), std::vector<uint32_t>(stops),
            std::move(arc_src), std::move(arc_lbl), std::move(arc_dst)};
}

PeekabooDecompResult DirtyPeekaboo::decompose(
    const std::vector<uint32_t>& target, bool minimize
) {
    uint16_t step_n = static_cast<uint16_t>(target.size());

    arena_ = PowersetArena();
    eps_cache_.clear();

    PeekabooNFA nfa(fst_, target.data(), static_cast<uint32_t>(target.size()),
                    step_n, &sym_to_idx_);

    // Initial state
    auto raw_starts = nfa.start_states();
    std::vector<uint64_t> init_closed;
    nfa.eps_closure_set(raw_starts, init_closed, eps_cache_);

    bool any_final = false;
    for (uint64_t s : init_closed) {
        if (nfa.is_final(s)) { any_final = true; break; }
    }
    uint32_t start_id = arena_.intern(std::move(init_closed), any_final);

    // BFS to build the full DFA
    std::deque<uint32_t> worklist;
    std::unordered_set<uint32_t> visited;
    worklist.push_back(start_id);
    visited.insert(start_id);

    std::vector<uint32_t> arc_src, arc_lbl, arc_dst;
    std::vector<uint32_t> reachable_order;  // BFS order

    size_t num_source_symbols = fst_->source_alphabet.size();
    std::unordered_set<uint16_t> relevant_y_idxs;

    while (!worklist.empty()) {
        uint32_t sid = worklist.front();
        worklist.pop_front();
        reachable_order.push_back(sid);

        // Collect relevant output symbols
        for (uint64_t packed : arena_.sets[sid]) {
            uint32_t fs; uint16_t bl, es; bool tr;
            unpack_peekaboo(packed, fs, bl, es, tr);
            if (fst_->is_final[fs] && bl == step_n + 1 && es != NO_EXTRA) {
                relevant_y_idxs.insert(es);
            }
        }

        // Expand arcs
        auto all_arcs = nfa.compute_all_arcs(arena_.sets[sid], eps_cache_);
        for (auto& [sym, successor] : all_arcs) {
            bool succ_final = false;
            for (uint64_t s : successor) {
                if (nfa.is_final(s)) { succ_final = true; break; }
            }
            uint32_t dest_id = arena_.intern(std::move(successor), succ_final);
            arc_src.push_back(sid);
            arc_lbl.push_back(sym);
            arc_dst.push_back(dest_id);

            if (visited.insert(dest_id).second) {
                worklist.push_back(dest_id);
            }
        }
    }

    // Build reachable flags
    uint32_t num_states = static_cast<uint32_t>(arena_.len());
    std::vector<bool> reachable(num_states, false);
    for (uint32_t sid : reachable_order) {
        reachable[sid] = true;
    }

    // Per-symbol Q/R classification with proper universality
    std::unordered_map<uint32_t, std::vector<uint32_t>> q_stops;
    std::unordered_map<uint32_t, std::vector<uint32_t>> r_stops;
    std::unordered_set<uint32_t> global_q_stops;

    for (uint16_t y_idx : relevant_y_idxs) {
        uint32_t sym = idx_to_sym_[y_idx];
        PerSymbolUnivFilter filter(fst_, step_n, y_idx);

        for (uint32_t sid = 0; sid < num_states; sid++) {
            if (!reachable[sid]) continue;
            if (!is_projected_final(arena_.sets[sid], y_idx, step_n, fst_)) continue;

            bool is_uni = filter.is_universal(
                arena_.sets[sid], y_idx, nfa, arena_, eps_cache_,
                num_source_symbols, step_n
            );

            if (is_uni) {
                q_stops[sym].push_back(sid);
                global_q_stops.insert(sid);
            } else {
                r_stops[sym].push_back(sid);
            }
        }
    }

    // Build per-symbol FSA results with backward trimming
    PeekabooDecompResult result;
    std::unordered_set<uint32_t> all_syms;
    for (auto& [sym, _] : q_stops) all_syms.insert(sym);
    for (auto& [sym, _] : r_stops) all_syms.insert(sym);

    for (uint32_t sym : all_syms) {
        PeekabooSymResult sr;
        sr.quotient = collect_arcs_trimmed(
            start_id, q_stops[sym], num_states,
            arc_src, arc_lbl, arc_dst, global_q_stops, reachable
        );
        sr.remainder = collect_arcs_trimmed(
            start_id, r_stops[sym], num_states,
            arc_src, arc_lbl, arc_dst, global_q_stops, reachable
        );
        result.symbols.push_back(sym);
        result.by_symbol[sym] = std::move(sr);
    }

    prev_target_len_ = step_n;
    return result;
}

}  // namespace transduction
