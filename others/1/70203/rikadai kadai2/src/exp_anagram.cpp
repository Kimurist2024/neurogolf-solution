#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static int add_hash(const std::string& key) {
    int h = 0;
    for (unsigned char c : key) {
        h += static_cast<int>(c);
    }
    return h % 17;
}

static int fnv1a_hash(const std::string& key) {
    std::uint32_t h = 2166136261u;
    for (unsigned char c : key) {
        h ^= static_cast<std::uint32_t>(c);
        h *= 16777619u;
    }
    return static_cast<int>(h % 17u);
}

int main() {
    const std::vector<std::vector<std::string>> groups = {
        {"abc", "acb", "bac", "bca", "cab", "cba"},
        {"stop", "tops", "pots", "spot", "opts", "post"},
        {"listen", "silent", "enlist", "tinsel"},
    };

    std::cout << std::left << std::setw(14) << "key"
              << std::right << std::setw(16) << "add_hash(m=17)"
              << std::setw(20) << "fnv1a_hash(m=17)" << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (const auto& group : groups) {
        for (const auto& key : group) {
            std::cout << std::left << std::setw(14) << key
                      << std::right << std::setw(16) << add_hash(key)
                      << std::setw(20) << fnv1a_hash(key) << "\n";
        }
        std::cout << std::string(50, '-') << "\n";
    }

    std::cout << "[まとめ] すべてのグループで add は全キーが同一バケットになり、"
              << "fnv1a は異なるバケットに分散している。\n";

    return 0;
}
