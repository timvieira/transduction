#include "peekaboo_fst.h"

#include <algorithm>
#include <deque>
#include <unordered_set>

namespace transduction {

std::vector<uint64_t> PeekabooNFA::start_states() const {
    std::vector<uint64_t> result;
    for (uint32_t s : fst->start_states) {
        result.push_back(pack_peekaboo(s, 0, NO_EXTRA, false));
    }
    return result;
}

bool PeekabooNFA::is_final(uint64_t packed) const {
    uint32_t fs; uint16_t bl, es; bool tr;
    unpack_peekaboo(packed, fs, bl, es, tr);
    return fst->is_final[fs] && bl == step_n + 1 && es != NO_EXTRA;
}

bool PeekabooNFA::is_productive(uint64_t packed) const {
    uint32_t fs; uint16_t bl, es; bool tr;
    unpack_peekaboo(packed, fs, bl, es, tr);

    if (fst->has_non_eps_input[fs]) return true;

    // NFA-final
    if (fst->is_final[fs] && bl == step_n + 1 && es != NO_EXTRA) return true;

    // Is-preimage
    if (fst->is_final[fs] && bl == step_n) {
        if (es == NO_EXTRA) return true;
        if (step_n > 0 && (step_n - 1) < full_target_len) {
            auto it = sym_to_idx->find(full_target[step_n - 1]);
            if (it != sym_to_idx->end() && es == it->second) return true;
        }
    }

    // Truncated
    if (tr) return true;

    return false;
}

PeekabooNFA::EffState PeekabooNFA::effective_state(
    uint16_t buf_len, uint16_t extra_sym, bool truncated
) const {
    if (extra_sym == NO_EXTRA) {
        return {buf_len, NO_EXTRA, true};
    }
    uint16_t prefix_len = buf_len - 1;
    if (prefix_len >= step_n) {
        return {buf_len, extra_sym, true};
    }
    // prefix_len < step_n: check match
    if (prefix_len < full_target_len) {
        auto it = sym_to_idx->find(full_target[prefix_len]);
        if (it != sym_to_idx->end() && extra_sym == it->second) {
            return {buf_len, NO_EXTRA, true};
        }
    }
    return {buf_len, extra_sym, false};
}

std::vector<std::pair<uint32_t, uint64_t>> PeekabooNFA::arcs(uint64_t packed) const {
    uint32_t i; uint16_t buf_len, extra_sym; bool truncated;
    unpack_peekaboo(packed, i, buf_len, extra_sym, truncated);

    auto [eff_n, eff_extra, is_valid] = effective_state(buf_len, extra_sym, truncated);
    if (!is_valid) return {};

    std::vector<std::pair<uint32_t, uint64_t>> result;

    if (eff_n >= step_n) {
        // Buffer has reached or passed step_n
        size_t count;
        auto* a = fst->arcs_from(i, count);
        for (size_t k = 0; k < count; k++) {
            uint32_t x = a[k].input;
            uint32_t y = a[k].output;
            uint32_t j = a[k].dest;

            if (y == INTERNAL_EPSILON || truncated) {
                result.emplace_back(x, pack_peekaboo(j, buf_len, extra_sym, truncated));
            } else if (eff_extra == NO_EXTRA && eff_n == step_n) {
                auto it = sym_to_idx->find(y);
                if (it != sym_to_idx->end()) {
                    result.emplace_back(x, pack_peekaboo(j, step_n + 1, it->second, false));
                }
            } else {
                result.emplace_back(x, pack_peekaboo(j, buf_len, extra_sym, true));
            }
        }
    } else {
        // Buffer hasn't reached step_n yet
        if (truncated) return {};
        // eff_extra must be NO_EXTRA
        size_t count;
        auto* a = fst->arcs_from(i, count);
        for (size_t k = 0; k < count; k++) {
            uint32_t x = a[k].input;
            uint32_t y = a[k].output;
            uint32_t j = a[k].dest;

            if (y == INTERNAL_EPSILON) {
                result.emplace_back(x, pack_peekaboo(j, eff_n, NO_EXTRA, false));
            } else if (y == full_target[eff_n]) {
                result.emplace_back(x, pack_peekaboo(j, eff_n + 1, NO_EXTRA, false));
            }
        }
    }
    return result;
}

std::shared_ptr<const std::vector<uint64_t>> PeekabooNFA::eps_closure_single(
    uint64_t state, EpsCache& cache
) const {
    auto it = cache.find(state);
    if (it != cache.end()) return it->second.first;  // O(1) shared_ptr copy

    std::unordered_set<uint64_t> visited;
    std::deque<uint64_t> worklist;
    visited.insert(state);
    worklist.push_back(state);

    while (!worklist.empty()) {
        uint64_t s = worklist.front();
        worklist.pop_front();
        for (auto& [x, dest] : arcs(s)) {
            if (x == INTERNAL_EPSILON && visited.insert(dest).second) {
                worklist.push_back(dest);
            }
        }
    }

    std::vector<uint64_t> result;
    for (uint64_t s : visited) {
        if (is_productive(s)) result.push_back(s);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());

    uint16_t max_bl = 0;
    for (uint64_t s : result) {
        uint16_t bl = static_cast<uint16_t>((s >> 17) & 0x7FFF);
        if (bl > max_bl) max_bl = bl;
    }

    auto rc = std::make_shared<const std::vector<uint64_t>>(std::move(result));
    cache[state] = {rc, max_bl};
    return rc;
}

void PeekabooNFA::eps_closure_set(
    const std::vector<uint64_t>& states, std::vector<uint64_t>& out,
    EpsCache& cache
) const {
    out.clear();
    for (uint64_t s : states) {
        auto closure = eps_closure_single(s, cache);
        out.insert(out.end(), closure->begin(), closure->end());
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

std::vector<std::pair<uint32_t, std::vector<uint64_t>>> PeekabooNFA::compute_all_arcs(
    const std::vector<uint64_t>& states, EpsCache& cache
) const {
    std::unordered_map<uint32_t, std::vector<uint64_t>> by_symbol;

    for (uint64_t packed : states) {
        for (auto& [x, dest] : arcs(packed)) {
            if (x != INTERNAL_EPSILON) {
                auto closure = eps_closure_single(dest, cache);
                auto& bucket = by_symbol[x];
                bucket.insert(bucket.end(), closure->begin(), closure->end());
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

void PeekabooNFA::compute_all_arcs_into(
    const std::vector<uint64_t>& states, EpsCache& cache,
    std::unordered_map<uint32_t, std::vector<uint64_t>>& buf
) const {
    // Clear values but keep entries (preserves allocated capacity)
    for (auto& [k, v] : buf) v.clear();

    for (uint64_t packed : states) {
        for (auto& [x, dest] : arcs(packed)) {
            if (x != INTERNAL_EPSILON) {
                auto closure = eps_closure_single(dest, cache);
                auto& bucket = buf[x];
                bucket.insert(bucket.end(), closure->begin(), closure->end());
            }
        }
    }

    // Sort and deduplicate each bucket in-place
    for (auto& [sym, v] : buf) {
        if (!v.empty()) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
    }
}

}  // namespace transduction
