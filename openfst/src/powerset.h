#pragma once
/// PowersetArena: hash-consing for sorted Vec<uint64_t> state sets.
/// Mirrors Rust powerset.rs.

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace transduction {

/// Hash for vector<uint64_t> (used as hash map key).
struct VecU64Hash {
    size_t operator()(const std::vector<uint64_t>& v) const {
        size_t seed = v.size();
        for (uint64_t x : v) {
            // FNV-1a inspired mixing
            seed ^= std::hash<uint64_t>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class PowersetArena {
public:
    std::unordered_map<std::vector<uint64_t>, uint32_t, VecU64Hash> map;
    std::unordered_map<uint64_t, uint32_t> single_map;  // Fast path for single-element sets
    std::vector<std::vector<uint64_t>> sets;
    std::vector<bool> is_final;

    PowersetArena() = default;

    /// Intern a sorted set of NFA states. Returns u32 ID.
    /// On cache hit, is_final is always updated.
    uint32_t intern(std::vector<uint64_t> sorted_set, bool any_final) {
        if (sorted_set.size() == 1) {
            uint64_t key = sorted_set[0];
            auto it = single_map.find(key);
            if (it != single_map.end()) {
                is_final[it->second] = any_final;
                return it->second;
            }
            uint32_t id = static_cast<uint32_t>(sets.size());
            sets.push_back(std::move(sorted_set));
            is_final.push_back(any_final);
            single_map[key] = id;
            return id;
        }

        auto it = map.find(sorted_set);
        if (it != map.end()) {
            is_final[it->second] = any_final;
            return it->second;
        }
        uint32_t id = static_cast<uint32_t>(sets.size());
        sets.push_back(sorted_set);
        is_final.push_back(any_final);
        map[sorted_set] = id;
        return id;
    }

    size_t len() const { return sets.size(); }
};

}  // namespace transduction
