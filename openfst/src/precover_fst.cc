#include "precover_fst.h"

#include <algorithm>
#include <deque>
#include <unordered_set>

namespace transduction {

std::vector<std::pair<uint32_t, uint64_t>> PrecoverNFA::arcs(uint64_t packed) const {
    auto [i, n] = unpack_state(packed);
    std::vector<std::pair<uint32_t, uint64_t>> result;

    if (n == target_len) {
        // Boundary phase: all arcs from FST state i
        size_t count;
        auto* a = fst->arcs_from(i, count);
        result.reserve(count);
        for (size_t k = 0; k < count; k++) {
            result.emplace_back(a[k].input, pack_state(a[k].dest, target_len));
        }
    } else {
        // Growing phase: arcs with output=EPSILON
        size_t count;
        auto* a = fst->arcs_by_output(i, INTERNAL_EPSILON, count);
        for (size_t k = 0; k < count; k++) {
            result.emplace_back(a[k].input, pack_state(a[k].dest, n));
        }
        // Growing phase: arcs with output=target[n]
        a = fst->arcs_by_output(i, target[n], count);
        for (size_t k = 0; k < count; k++) {
            result.emplace_back(a[k].input, pack_state(a[k].dest, n + 1));
        }
    }
    return result;
}

std::vector<uint64_t> PrecoverNFA::arcs_eps(uint64_t packed) const {
    auto [i, n] = unpack_state(packed);
    std::vector<uint64_t> result;

    if (n == target_len) {
        size_t count;
        auto* ea = fst->eps_input_arcs(i, count);
        result.reserve(count);
        for (size_t k = 0; k < count; k++) {
            result.push_back(pack_state(ea[k].dest, target_len));
        }
    } else {
        // eps-input with output=EPSILON
        size_t count;
        auto* ea = fst->eps_input_arcs_by_output(i, INTERNAL_EPSILON, count);
        for (size_t k = 0; k < count; k++) {
            result.push_back(pack_state(ea[k].dest, n));
        }
        // eps-input with output=target[n]
        ea = fst->eps_input_arcs_by_output(i, target[n], count);
        for (size_t k = 0; k < count; k++) {
            result.push_back(pack_state(ea[k].dest, n + 1));
        }
    }
    return result;
}

std::vector<uint64_t> PrecoverNFA::start_states() const {
    std::vector<uint64_t> result;
    result.reserve(fst->start_states.size());
    for (uint32_t s : fst->start_states) {
        result.push_back(pack_state(s, 0));
    }
    return result;
}

bool PrecoverNFA::is_productive(uint64_t packed) const {
    auto [fst_state, buf_pos] = unpack_state(packed);
    return fst->has_non_eps_input[fst_state]
        || (fst->is_final[fst_state] && buf_pos == target_len);
}

const std::vector<uint64_t>& PrecoverNFA::eps_closure_single(uint64_t state) {
    auto it = eps_cache.find(state);
    if (it != eps_cache.end()) {
        eps_cache_hits++;
        return it->second.first;
    }
    eps_cache_misses++;

    // BFS
    std::unordered_set<uint64_t> visited;
    std::deque<uint64_t> worklist;
    visited.insert(state);
    worklist.push_back(state);

    while (!worklist.empty()) {
        uint64_t s = worklist.front();
        worklist.pop_front();
        for (uint64_t dest : arcs_eps(s)) {
            if (visited.insert(dest).second) {
                worklist.push_back(dest);
            }
        }
    }

    // Filter to productive
    std::vector<uint64_t> result;
    for (uint64_t s : visited) {
        if (is_productive(s)) {
            result.push_back(s);
        }
    }
    std::sort(result.begin(), result.end());

    // max_buf_pos
    uint32_t max_bp = 0;
    for (uint64_t s : result) {
        uint32_t bp = static_cast<uint32_t>(s % stride);
        if (bp > max_bp) max_bp = bp;
    }

    auto& entry = eps_cache[state];
    entry = {std::move(result), max_bp};
    return entry.first;
}

void PrecoverNFA::eps_closure_set(const std::vector<uint64_t>& states, std::vector<uint64_t>& out) {
    out.clear();
    for (uint64_t s : states) {
        const auto& closure = eps_closure_single(s);
        out.insert(out.end(), closure.begin(), closure.end());
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

std::vector<std::pair<uint32_t, std::vector<uint64_t>>> PrecoverNFA::compute_all_arcs(
    const std::vector<uint64_t>& states
) {
    std::unordered_map<uint32_t, std::vector<uint64_t>> by_symbol;

    for (uint64_t packed : states) {
        auto arc_list = arcs(packed);
        for (auto& [x, dest] : arc_list) {
            if (x != INTERNAL_EPSILON) {
                const auto& closure = eps_closure_single(dest);
                auto& bucket = by_symbol[x];
                bucket.insert(bucket.end(), closure.begin(), closure.end());
            }
        }
    }

    std::vector<std::pair<uint32_t, std::vector<uint64_t>>> result;
    for (auto& [sym, v] : by_symbol) {
        if (!v.empty()) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            result.emplace_back(sym, std::move(v));
        }
    }
    return result;
}

}  // namespace transduction
