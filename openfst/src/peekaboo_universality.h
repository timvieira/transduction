#pragma once
/// Per-symbol universality filter for peekaboo DFA states.
/// Shared between DirtyPeekaboo and LazyPeekabooDFA.

#include "fst_convert.h"
#include "peekaboo_fst.h"
#include "powerset.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace transduction {

/// Project a full DFA state's NFA set to elements compatible with output symbol y_idx.
inline std::vector<uint64_t> project_and_refine(
    const std::vector<uint64_t>& full_nfa_set,
    uint16_t y_idx, uint16_t step_n
) {
    std::vector<uint64_t> projected;
    uint16_t target_len = step_n + 1;

    for (uint64_t packed : full_nfa_set) {
        uint32_t fs; uint16_t bl, es; bool tr;
        unpack_peekaboo(packed, fs, bl, es, tr);

        if (es == NO_EXTRA) {
            uint16_t clipped = std::min(bl, target_len);
            projected.push_back(pack_peekaboo(fs, clipped, NO_EXTRA, tr));
        } else if (es == y_idx) {
            uint16_t clipped = std::min(bl, target_len);
            projected.push_back(pack_peekaboo(fs, clipped, es, tr));
        }
    }

    std::sort(projected.begin(), projected.end());
    projected.erase(std::unique(projected.begin(), projected.end()), projected.end());
    return projected;
}

inline bool is_projected_final(
    const std::vector<uint64_t>& nfa_set,
    uint16_t y_idx, uint16_t step_n,
    const FstData* fst
) {
    for (uint64_t packed : nfa_set) {
        uint32_t fs; uint16_t bl, es; bool tr;
        unpack_peekaboo(packed, fs, bl, es, tr);
        if (fst->is_final[fs] && bl == step_n + 1 && es == y_idx) return true;
    }
    return false;
}

/// Per-symbol universality filter with witness/pos/neg caches and BFS fallback.
class PerSymbolUnivFilter {
public:
    std::unordered_set<uint64_t> witnesses;
    std::unordered_map<uint64_t, std::vector<uint32_t>> pos_index;
    std::vector<size_t> pos_sizes;
    std::unordered_map<uint64_t, std::vector<uint32_t>> neg_index;
    uint32_t neg_next = 0;

    PerSymbolUnivFilter(const FstData* fst, uint16_t step_n, uint16_t y_idx) {
        for (size_t q = 0; q < fst->ip_universal.size(); q++) {
            if (fst->ip_universal[q]) {
                witnesses.insert(pack_peekaboo(static_cast<uint32_t>(q), step_n + 1, y_idx, false));
            }
        }
    }

    void add_pos(const std::vector<uint64_t>& nfa_set) {
        uint32_t eid = static_cast<uint32_t>(pos_sizes.size());
        pos_sizes.push_back(nfa_set.size());
        for (uint64_t e : nfa_set) pos_index[e].push_back(eid);
    }

    void add_neg(const std::vector<uint64_t>& nfa_set) {
        uint32_t eid = neg_next++;
        for (uint64_t e : nfa_set) neg_index[e].push_back(eid);
    }

    bool has_pos_subset(const std::vector<uint64_t>& nfa_set) const {
        std::unordered_map<uint32_t, size_t> hits;
        for (uint64_t e : nfa_set) {
            auto it = pos_index.find(e);
            if (it != pos_index.end()) {
                for (uint32_t eid : it->second) {
                    size_t& h = hits[eid];
                    h++;
                    if (h == pos_sizes[eid]) return true;
                }
            }
        }
        return false;
    }

    bool has_neg_superset(const std::vector<uint64_t>& nfa_set) const {
        if (nfa_set.empty()) return neg_next > 0;
        size_t target_count = nfa_set.size();
        std::unordered_map<uint32_t, size_t> hits;
        for (uint64_t e : nfa_set) {
            auto it = neg_index.find(e);
            if (it == neg_index.end()) return false;
            for (uint32_t eid : it->second) {
                size_t& h = hits[eid];
                h++;
                if (h == target_count) return true;
            }
        }
        return false;
    }

    bool bfs_universal(
        const std::vector<uint64_t>& projected_set,
        uint16_t y_idx,
        const PeekabooNFA& nfa,
        PowersetArena& arena,
        EpsCache& eps_cache,
        size_t num_source_symbols,
        uint16_t step_n
    ) {
        bool any_final = false;
        for (uint64_t s : projected_set) {
            if (nfa.is_final(s)) { any_final = true; break; }
        }
        uint32_t start_id = arena.intern(std::vector<uint64_t>(projected_set), any_final);

        if (!any_final) return false;

        std::unordered_set<uint32_t> sub_visited;
        std::deque<uint32_t> sub_worklist;
        sub_visited.insert(start_id);
        sub_worklist.push_back(start_id);

        while (!sub_worklist.empty()) {
            uint32_t cur = sub_worklist.front();
            sub_worklist.pop_front();

            auto cur_set = arena.sets[cur];  // copy

            bool cur_final = false;
            for (uint64_t s : cur_set) {
                if (nfa.is_final(s)) { cur_final = true; break; }
            }
            if (!cur_final) return false;

            auto all_arcs = nfa.compute_all_arcs(cur_set, eps_cache);
            if (all_arcs.size() < num_source_symbols) return false;

            for (auto& [sym, successor] : all_arcs) {
                auto projected_succ = project_and_refine(successor, y_idx, step_n);
                if (projected_succ.empty()) return false;

                bool succ_final = false;
                for (uint64_t s : projected_succ) {
                    if (nfa.is_final(s)) { succ_final = true; break; }
                }
                uint32_t dest_id = arena.intern(std::move(projected_succ), succ_final);
                if (sub_visited.insert(dest_id).second) {
                    sub_worklist.push_back(dest_id);
                }
            }
        }
        return true;
    }

    bool is_universal(
        const std::vector<uint64_t>& full_nfa_set,
        uint16_t y_idx,
        const PeekabooNFA& nfa,
        PowersetArena& arena,
        EpsCache& eps_cache,
        size_t num_source_symbols,
        uint16_t step_n
    ) {
        auto projected = project_and_refine(full_nfa_set, y_idx, step_n);
        if (projected.empty()) return false;

        bool any_final = false;
        for (uint64_t packed : projected) {
            uint32_t fs; uint16_t bl, es; bool tr;
            unpack_peekaboo(packed, fs, bl, es, tr);
            if (nfa.fst->is_final[fs] && bl == step_n + 1 && es == y_idx) {
                any_final = true; break;
            }
        }
        if (!any_final) return false;

        for (uint64_t e : projected) {
            if (witnesses.count(e)) {
                add_pos(projected);
                return true;
            }
        }

        if (has_pos_subset(projected)) return true;
        if (has_neg_superset(projected)) return false;

        auto projected_copy = projected;
        bool result = bfs_universal(projected, y_idx, nfa, arena, eps_cache, num_source_symbols, step_n);
        if (result) add_pos(projected_copy);
        else add_neg(projected_copy);
        return result;
    }
};

}  // namespace transduction
