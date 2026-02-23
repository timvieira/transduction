#include "decompose.h"
#include "precover_fst.h"
#include "powerset.h"

#include <chrono>
#include <deque>
#include <unordered_set>

namespace transduction {

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(std::chrono::time_point<Clock> start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

DecompResult decompose(const FstData& fst, const uint32_t* target, uint32_t target_len) {
    auto total_start = Clock::now();
    ProfileStats stats;

    auto init_start = Clock::now();

    PrecoverNFA nfa(&fst, target, target_len);
    PowersetArena arena;
    size_t num_source_symbols = fst.source_alphabet.size();

    // 1. Epsilon-closed initial powerset state
    auto raw_starts = nfa.start_states();
    std::vector<uint64_t> init_closed;
    nfa.eps_closure_set(raw_starts, init_closed);

    bool any_final = false;
    for (uint64_t s : init_closed) {
        if (nfa.is_final(s)) { any_final = true; break; }
    }
    uint32_t start_id = arena.intern(std::move(init_closed), any_final);

    std::deque<uint32_t> worklist;
    std::unordered_set<uint32_t> visited;

    std::vector<uint32_t> arc_src, arc_lbl, arc_dst;
    std::vector<uint32_t> q_stop, r_stop;

    UniversalityFilter filter(fst.ip_universal, target_len, nfa.stride);

    worklist.push_back(start_id);
    visited.insert(start_id);

    stats.init_ms = elapsed_ms(init_start);

    auto bfs_start = Clock::now();

    // 2. BFS
    while (!worklist.empty()) {
        uint32_t sid = worklist.front();
        worklist.pop_front();

        if (arena.is_final[sid]) {
            auto uni_start = Clock::now();
            stats.universal_calls++;
            bool is_uni = filter.is_universal(sid, nfa, arena, num_source_symbols, stats);
            stats.universal_ms += elapsed_ms(uni_start);

            if (is_uni) {
                stats.universal_true++;
                q_stop.push_back(sid);
                continue;
            } else {
                stats.universal_false++;
                r_stop.push_back(sid);
            }
        }

        // Track powerset sizes
        size_t pset_size = arena.sets[sid].size();
        if (pset_size > stats.max_powerset_size) {
            stats.max_powerset_size = pset_size;
        }

        auto arcs_start = Clock::now();
        auto all_arcs = nfa.compute_all_arcs(arena.sets[sid]);
        stats.compute_arcs_ms += elapsed_ms(arcs_start);
        stats.compute_arcs_calls++;

        for (auto& [x, successor] : all_arcs) {
            auto intern_start = Clock::now();
            bool succ_final = false;
            for (uint64_t s : successor) {
                if (nfa.is_final(s)) { succ_final = true; break; }
            }
            uint32_t dest_id = arena.intern(std::move(successor), succ_final);
            stats.intern_ms += elapsed_ms(intern_start);
            stats.intern_calls++;

            arc_src.push_back(sid);
            arc_lbl.push_back(x);
            arc_dst.push_back(dest_id);

            if (visited.insert(dest_id).second) {
                worklist.push_back(dest_id);
            }
        }
    }

    stats.bfs_ms = elapsed_ms(bfs_start);
    stats.eps_cache_hits = nfa.eps_cache_hits;
    stats.eps_cache_misses = nfa.eps_cache_misses;

    // Compute avg powerset size
    size_t total_pset = 0;
    for (auto& s : arena.sets) total_pset += s.size();
    stats.avg_powerset_size = arena.len() > 0
        ? static_cast<double>(total_pset) / arena.len()
        : 0.0;

    // 3. Build Q and R
    uint32_t num_states = static_cast<uint32_t>(arena.len());
    stats.dfa_states = num_states;
    stats.total_arcs = arc_src.size();
    stats.q_stops = static_cast<uint32_t>(q_stop.size());
    stats.r_stops = static_cast<uint32_t>(r_stop.size());
    stats.total_ms = elapsed_ms(total_start);

    FsaResult quotient{num_states, {start_id}, std::move(q_stop),
                       arc_src, arc_lbl, arc_dst};
    FsaResult remainder{num_states, {start_id}, std::move(r_stop),
                        std::move(arc_src), std::move(arc_lbl), std::move(arc_dst)};

    return {std::move(quotient), std::move(remainder), stats};
}

}  // namespace transduction
