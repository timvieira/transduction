#include "universality.h"

#include <deque>
#include <unordered_set>

namespace transduction {

UniversalityFilter::UniversalityFilter(
    const std::vector<bool>& ip_univ, uint32_t target_len, uint64_t stride
) {
    for (size_t q = 0; q < ip_univ.size(); q++) {
        if (ip_univ[q]) {
            witnesses.insert(static_cast<uint64_t>(q) * stride + target_len);
        }
    }
}

void UniversalityFilter::add_pos(const std::vector<uint64_t>& nfa_set) {
    uint32_t eid = static_cast<uint32_t>(pos_sizes.size());
    pos_sizes.push_back(nfa_set.size());
    for (uint64_t e : nfa_set) {
        pos_index[e].push_back(eid);
    }
}

void UniversalityFilter::add_neg(const std::vector<uint64_t>& nfa_set) {
    uint32_t eid = neg_next++;
    for (uint64_t e : nfa_set) {
        neg_index[e].push_back(eid);
    }
}

bool UniversalityFilter::has_pos_subset(const std::vector<uint64_t>& nfa_set) const {
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

bool UniversalityFilter::has_neg_superset(const std::vector<uint64_t>& nfa_set) const {
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

bool UniversalityFilter::bfs_universal(
    uint32_t sid,
    PrecoverNFA& nfa,
    PowersetArena& arena,
    size_t num_source_symbols,
    ProfileStats& stats
) {
    // Check finality from NFA
    bool has_final = false;
    for (uint64_t s : arena.sets[sid]) {
        if (nfa.is_final(s)) { has_final = true; break; }
    }
    if (!has_final) return false;

    std::unordered_set<uint32_t> sub_visited;
    std::deque<uint32_t> sub_worklist;
    sub_visited.insert(sid);
    sub_worklist.push_back(sid);

    while (!sub_worklist.empty()) {
        uint32_t cur = sub_worklist.front();
        sub_worklist.pop_front();

        // Check finality
        bool cur_final = false;
        for (uint64_t s : arena.sets[cur]) {
            if (nfa.is_final(s)) { cur_final = true; break; }
        }
        if (!cur_final) return false;

        stats.universal_sub_bfs_states++;

        auto all_arcs = nfa.compute_all_arcs(arena.sets[cur]);
        stats.universal_compute_arcs_calls++;

        if (all_arcs.size() < num_source_symbols) return false;

        for (auto& [sym, successor] : all_arcs) {
            bool any_final = false;
            for (uint64_t s : successor) {
                if (nfa.is_final(s)) { any_final = true; break; }
            }
            uint32_t dest_id = arena.intern(std::move(successor), any_final);
            if (sub_visited.insert(dest_id).second) {
                sub_worklist.push_back(dest_id);
            }
        }
    }
    return true;
}

bool UniversalityFilter::is_universal(
    uint32_t sid,
    PrecoverNFA& nfa,
    PowersetArena& arena,
    size_t num_source_symbols,
    ProfileStats& stats
) {
    const auto& nfa_set = arena.sets[sid];

    // 1. Witness check
    for (uint64_t e : nfa_set) {
        if (witnesses.count(e)) {
            add_pos(nfa_set);
            return true;
        }
    }

    // 2. Superset monotonicity
    if (has_pos_subset(nfa_set)) return true;

    // 3. Subset monotonicity
    if (has_neg_superset(nfa_set)) return false;

    // 4. BFS fallback
    auto nfa_set_copy = nfa_set;
    bool result = bfs_universal(sid, nfa, arena, num_source_symbols, stats);
    if (result) {
        add_pos(nfa_set_copy);
    } else {
        add_neg(nfa_set_copy);
    }
    return result;
}

}  // namespace transduction
