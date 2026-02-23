#include "fst_convert.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>

namespace transduction {

// ---------------------------------------------------------------------------
// compute_ip_universal_states (greatest fixpoint, mirrors Rust fst.rs)
// ---------------------------------------------------------------------------

static std::vector<uint32_t> ip_eps_close(
    const std::vector<uint32_t>& states, const FstData& fst
) {
    std::unordered_set<uint32_t> visited(states.begin(), states.end());
    std::deque<uint32_t> worklist(states.begin(), states.end());
    while (!worklist.empty()) {
        uint32_t s = worklist.front();
        worklist.pop_front();
        size_t count;
        auto* ea = fst.eps_input_arcs(s, count);
        for (size_t i = 0; i < count; i++) {
            if (visited.insert(ea[i].dest).second) {
                worklist.push_back(ea[i].dest);
            }
        }
    }
    std::vector<uint32_t> result(visited.begin(), visited.end());
    std::sort(result.begin(), result.end());
    return result;
}

static std::vector<bool> compute_ip_universal_states(const FstData& fst) {
    uint32_t n = fst.num_states;
    const auto& sa = fst.source_alphabet;

    if (sa.empty()) {
        std::vector<bool> result(n, false);
        for (uint32_t q = 0; q < n; q++) {
            auto closure = ip_eps_close({q}, fst);
            for (uint32_t s : closure) {
                if (fst.is_final[s]) { result[q] = true; break; }
            }
        }
        return result;
    }

    // Precompute closures
    std::vector<std::vector<uint32_t>> closures(n);
    for (uint32_t q = 0; q < n; q++) {
        closures[q] = ip_eps_close({q}, fst);
    }

    // by_symbol[q][sym] = set of raw destinations
    std::vector<std::unordered_map<uint32_t, std::unordered_set<uint32_t>>> by_symbol(n);
    for (uint32_t q = 0; q < n; q++) {
        for (uint32_t s : closures[q]) {
            size_t count;
            auto* a = fst.arcs_from(s, count);
            for (size_t i = 0; i < count; i++) {
                if (a[i].input != INTERNAL_EPSILON) {
                    by_symbol[q][a[i].input].insert(a[i].dest);
                }
            }
        }
    }

    // Greatest fixpoint
    std::vector<bool> candidates(n, true);
    bool changed = true;
    while (changed) {
        changed = false;
        for (uint32_t q = 0; q < n; q++) {
            if (!candidates[q]) continue;

            // Must contain a final state
            bool has_final = false;
            for (uint32_t s : closures[q]) {
                if (fst.is_final[s]) { has_final = true; break; }
            }
            if (!has_final) {
                candidates[q] = false; changed = true; continue;
            }

            // Must be complete on source alphabet
            bool complete = true;
            for (uint32_t a : sa) {
                if (by_symbol[q].find(a) == by_symbol[q].end()) {
                    complete = false; break;
                }
            }
            if (!complete) {
                candidates[q] = false; changed = true; continue;
            }

            // For each symbol, successor closure must contain a candidate
            bool ok = true;
            for (uint32_t a : sa) {
                auto it = by_symbol[q].find(a);
                if (it == by_symbol[q].end()) { ok = false; break; }
                std::vector<uint32_t> raw(it->second.begin(), it->second.end());
                auto succ_closure = ip_eps_close(raw, fst);
                bool has_candidate = false;
                for (uint32_t s : succ_closure) {
                    if (candidates[s]) { has_candidate = true; break; }
                }
                if (!has_candidate) { ok = false; break; }
            }
            if (!ok) {
                candidates[q] = false; changed = true;
            }
        }
    }
    return candidates;
}

// ---------------------------------------------------------------------------
// make_fst_data
// ---------------------------------------------------------------------------

std::unique_ptr<FstData> make_fst_data(
    uint32_t num_states,
    const std::vector<uint32_t>& start_states,
    const std::vector<uint32_t>& final_states,
    const std::vector<uint32_t>& arc_src,
    const std::vector<uint32_t>& arc_in,
    const std::vector<uint32_t>& arc_out,
    const std::vector<uint32_t>& arc_dst,
    const std::vector<uint32_t>& source_alphabet
) {
    auto fst = std::make_unique<FstData>();
    fst->num_states = num_states;
    fst->start_states = start_states;
    fst->source_alphabet = source_alphabet;

    uint32_t n = num_states;
    size_t num_arcs = arc_src.size();

    // is_final
    fst->is_final.resize(n, false);
    for (uint32_t s : final_states) {
        fst->is_final[s] = true;
    }

    // Sort arcs by (src, output, input)
    std::vector<size_t> indices(num_arcs);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        if (arc_src[a] != arc_src[b]) return arc_src[a] < arc_src[b];
        if (arc_out[a] != arc_out[b]) return arc_out[a] < arc_out[b];
        return arc_in[a] < arc_in[b];
    });

    // Build CSR
    fst->offsets.resize(n + 1, 0);
    fst->arcs.resize(num_arcs);
    for (size_t k = 0; k < num_arcs; k++) {
        size_t idx = indices[k];
        fst->offsets[arc_src[idx] + 1]++;
        fst->arcs[k] = {arc_in[idx], arc_out[idx], arc_dst[idx]};
    }
    for (uint32_t i = 1; i <= n; i++) {
        fst->offsets[i] += fst->offsets[i - 1];
    }

    // Build output-group directory + epsilon side table
    fst->group_offsets.resize(n + 1, 0);
    fst->eps_offsets.resize(n + 1, 0);
    fst->has_non_eps_input.resize(n, false);

    for (uint32_t state = 0; state < n; state++) {
        uint32_t lo = fst->offsets[state];
        uint32_t hi = fst->offsets[state + 1];

        fst->group_offsets[state] = static_cast<uint32_t>(fst->output_groups.size());
        fst->eps_offsets[state] = static_cast<uint32_t>(fst->eps_arcs.size());

        if (lo < hi) {
            uint32_t prev_output = fst->arcs[lo].output;
            for (uint32_t pos = lo; pos < hi; pos++) {
                auto& arc = fst->arcs[pos];
                if (arc.output != prev_output) {
                    fst->output_groups.push_back({prev_output, pos});
                    prev_output = arc.output;
                }
                if (arc.input == INTERNAL_EPSILON) {
                    fst->eps_arcs.push_back({arc.output, arc.dest});
                } else {
                    fst->has_non_eps_input[state] = true;
                }
            }
            fst->output_groups.push_back({prev_output, hi});
        }
    }
    fst->group_offsets[n] = static_cast<uint32_t>(fst->output_groups.size());
    fst->eps_offsets[n] = static_cast<uint32_t>(fst->eps_arcs.size());

    // Compute ip-universal states
    fst->ip_universal = compute_ip_universal_states(*fst);

    return fst;
}

// ---------------------------------------------------------------------------
// make_vector_fst (for reference; not used in main decomposition path)
// ---------------------------------------------------------------------------

std::unique_ptr<fst::StdVectorFst> make_vector_fst(
    uint32_t num_states,
    const std::vector<uint32_t>& start_states,
    const std::vector<uint32_t>& final_states,
    const std::vector<uint32_t>& arc_src,
    const std::vector<uint32_t>& arc_in,
    const std::vector<uint32_t>& arc_out,
    const std::vector<uint32_t>& arc_dst
) {
    auto vfst = std::make_unique<fst::StdVectorFst>();

    for (uint32_t i = 0; i < num_states; i++) {
        vfst->AddState();
    }

    if (!start_states.empty()) {
        vfst->SetStart(start_states[0]);
    }

    for (uint32_t s : final_states) {
        vfst->SetFinal(s, fst::StdArc::Weight::One());
    }

    for (size_t i = 0; i < arc_src.size(); i++) {
        // Remap: INTERNAL_EPSILON -> 0, else label+1
        int32_t ilabel = (arc_in[i] == INTERNAL_EPSILON) ? 0 : static_cast<int32_t>(arc_in[i] + 1);
        int32_t olabel = (arc_out[i] == INTERNAL_EPSILON) ? 0 : static_cast<int32_t>(arc_out[i] + 1);
        vfst->AddArc(arc_src[i], fst::StdArc(ilabel, olabel, fst::StdArc::Weight::One(), arc_dst[i]));
    }

    return vfst;
}

}  // namespace transduction
