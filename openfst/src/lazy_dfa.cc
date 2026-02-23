#include "lazy_dfa.h"

#include <algorithm>
#include <deque>
#include <unordered_set>

namespace transduction {

LazyPeekabooDFA::LazyPeekabooDFA(const FstData* fst) : fst_(fst) {
    // Build symbol index from source + target alphabets (all non-eps labels in arcs)
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

void LazyPeekabooDFA::new_step(const std::vector<uint32_t>& target) {
    current_target_ = target;
    arena_ = PowersetArena();
    eps_cache_.clear();
    arc_cache_.clear();
    classify_cache_.clear();
    start_ids_.clear();
    univ_filters_.clear();

    uint16_t step_n = static_cast<uint16_t>(target.size());

    nfa_.emplace(fst_, current_target_.data(),
                 static_cast<uint32_t>(current_target_.size()),
                 step_n, &sym_to_idx_);

    auto raw_starts = nfa_->start_states();
    std::vector<uint64_t> init_closed;
    nfa_->eps_closure_set(raw_starts, init_closed, eps_cache_);

    bool any_final = false;
    for (uint64_t s : init_closed) {
        if (nfa_->is_final(s)) { any_final = true; break; }
    }
    uint32_t sid = arena_.intern(std::move(init_closed), any_final);
    start_ids_.push_back(sid);
}

std::vector<uint32_t> LazyPeekabooDFA::start_ids() const {
    return start_ids_;
}

void LazyPeekabooDFA::ensure_arcs(uint32_t sid) {
    if (arc_cache_.count(sid)) return;

    nfa_->compute_all_arcs_into(arena_.sets[sid], eps_cache_, arcs_buf_);

    std::vector<std::pair<uint32_t, uint32_t>> result;
    for (auto& [sym, successor] : arcs_buf_) {
        if (successor.empty()) continue;
        bool succ_final = false;
        for (uint64_t s : successor) {
            if (nfa_->is_final(s)) { succ_final = true; break; }
        }
        uint32_t dest_id = arena_.intern(std::vector<uint64_t>(successor), succ_final);
        result.emplace_back(sym, dest_id);
    }
    arc_cache_[sid] = std::move(result);
}

std::vector<std::pair<uint32_t, uint32_t>> LazyPeekabooDFA::arcs(uint32_t sid) {
    ensure_arcs(sid);
    return arc_cache_[sid];
}

std::optional<uint32_t> LazyPeekabooDFA::run(const std::vector<uint32_t>& source_path) {
    if (start_ids_.empty()) return std::nullopt;
    uint32_t current = start_ids_[0];
    for (uint32_t x : source_path) {
        ensure_arcs(current);
        bool found = false;
        for (auto& [lbl, dst] : arc_cache_[current]) {
            if (lbl == x) {
                current = dst;
                found = true;
                break;
            }
        }
        if (!found) return std::nullopt;
    }
    return current;
}

ClassifyResult LazyPeekabooDFA::classify(uint32_t sid) {
    auto cache_it = classify_cache_.find(sid);
    if (cache_it != classify_cache_.end()) return cache_it->second;

    uint16_t step_n = static_cast<uint16_t>(current_target_.size());
    const auto& nfa_set = arena_.sets[sid];
    size_t num_source_symbols = fst_->source_alphabet.size();
    const auto& nfa = nfa_.value();

    ClassifyResult cr;
    cr.quotient_sym = -1;
    cr.is_preimage = false;
    cr.has_truncated = false;

    std::unordered_set<uint32_t> r_syms_set;
    std::unordered_set<uint32_t> trunc_syms_set;

    // Collect relevant y_idx values and metadata
    std::unordered_set<uint16_t> relevant_y_idxs;

    for (uint64_t packed : nfa_set) {
        uint32_t fs; uint16_t bl, es; bool tr;
        unpack_peekaboo(packed, fs, bl, es, tr);

        // Truncated
        if (tr) {
            cr.has_truncated = true;
            if (es != NO_EXTRA && es < idx_to_sym_.size()) {
                trunc_syms_set.insert(idx_to_sym_[es]);
            }
        }

        // Is-preimage: fst-final with buffer == target (buf_len == step_n, on-target)
        if (fst_->is_final[fs] && bl == step_n && es == NO_EXTRA) {
            cr.is_preimage = true;
        }
        // Non-canonical preimage (off-target matching target[step_n-1])
        if (fst_->is_final[fs] && bl == step_n && es != NO_EXTRA && step_n > 0) {
            if ((step_n - 1) < current_target_.size()) {
                auto it = sym_to_idx_.find(current_target_[step_n - 1]);
                if (it != sym_to_idx_.end() && es == it->second) {
                    cr.is_preimage = true;
                }
            }
        }

        // NFA-final: contributes to Q or R for the extra symbol
        if (fst_->is_final[fs] && bl == step_n + 1 && es != NO_EXTRA) {
            relevant_y_idxs.insert(es);
        }
    }

    // Per-symbol universality classification using PerSymbolUnivFilter
    // Process symbols in sorted order (matching Rust behavior: first universal = quotient)
    std::vector<uint16_t> sorted_y_idxs(relevant_y_idxs.begin(), relevant_y_idxs.end());
    std::sort(sorted_y_idxs.begin(), sorted_y_idxs.end());

    for (uint16_t y_idx : sorted_y_idxs) {
        uint32_t sym = idx_to_sym_[y_idx];

        // Check if this DFA state has projected-final elements for y
        if (!is_projected_final(nfa_set, y_idx, step_n, fst_)) {
            // Has relevant elements but not projected-final -> remainder
            r_syms_set.insert(sym);
            continue;
        }

        // Get or create universality filter for this symbol
        auto filter_it = univ_filters_.find(y_idx);
        if (filter_it == univ_filters_.end()) {
            filter_it = univ_filters_.emplace(y_idx,
                PerSymbolUnivFilter(fst_, step_n, y_idx)).first;
        }

        bool is_uni = filter_it->second.is_universal(
            nfa_set, y_idx, nfa, arena_, eps_cache_,
            num_source_symbols, step_n
        );

        if (is_uni) {
            if (cr.quotient_sym < 0) {
                cr.quotient_sym = static_cast<int32_t>(sym);
            } else {
                // Already have a quotient symbol; additional universals go to remainder
                r_syms_set.insert(sym);
            }
        } else {
            r_syms_set.insert(sym);
        }
    }

    cr.remainder_syms.assign(r_syms_set.begin(), r_syms_set.end());
    cr.trunc_output_syms.assign(trunc_syms_set.begin(), trunc_syms_set.end());

    classify_cache_[sid] = cr;
    return cr;
}

std::vector<ExpandEntry> LazyPeekabooDFA::expand_batch(
    const std::vector<uint32_t>& sids
) {
    std::vector<ExpandEntry> result;
    result.reserve(sids.size());
    for (uint32_t sid : sids) {
        ExpandEntry entry;
        entry.sid = sid;
        entry.cr = classify(sid);  // uses classify_cache_
        if (entry.cr.quotient_sym < 0) {
            ensure_arcs(sid);
            entry.arcs = arc_cache_[sid];  // copy (small: just u32 pairs)
        }
        result.push_back(std::move(entry));
    }
    return result;
}

std::vector<std::tuple<uint32_t, uint16_t, uint16_t, bool>>
LazyPeekabooDFA::decode_state(uint32_t sid) const {
    std::vector<std::tuple<uint32_t, uint16_t, uint16_t, bool>> result;
    for (uint64_t packed : arena_.sets[sid]) {
        uint32_t fs; uint16_t bl, es; bool tr;
        unpack_peekaboo(packed, fs, bl, es, tr);
        result.emplace_back(fs, bl, es, tr);
    }
    return result;
}

std::vector<uint32_t> LazyPeekabooDFA::idx_to_sym_map() const {
    return idx_to_sym_;
}

}  // namespace transduction
