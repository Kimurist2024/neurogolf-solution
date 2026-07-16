#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static int add_hash(const std::string& key, int m) {
    int h = 0;
    for (unsigned char c : key) {
        h += static_cast<int>(c);
    }
    return h % m;
}

static std::uint32_t fnv1a_hash_raw(const std::string& key) {
    std::uint32_t h = 2166136261u;
    for (unsigned char c : key) {
        h ^= static_cast<std::uint32_t>(c);
        h *= 16777619u;
    }
    return h;
}

static int fnv1a_hash(const std::string& key, int m) {
    return static_cast<int>(fnv1a_hash_raw(key) % static_cast<std::uint32_t>(m));
}

struct Stats {
    std::string name;
    std::vector<int> counts;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <keyfile>\n";
        return 1;
    }

    const std::string keyfile = argv[1];
    std::ifstream ifs(keyfile);
    if (!ifs) {
        std::cerr << "cannot open " << keyfile << "\n";
        return 1;
    }

    std::vector<std::string> keys;
    std::string line;
    while (std::getline(ifs, line)) {
        keys.push_back(line);
    }

    const int n = static_cast<int>(keys.size());
    const std::vector<int> ms = {16, 17, 256, 257, 1024, 1031};

    std::cout << "hash,m,n,empty,longest,mean_nonempty,variance,chi2\n";

    for (int m : ms) {
        std::vector<Stats> stats = {
            {"add",   std::vector<int>(m, 0)},
            {"fnv1a", std::vector<int>(m, 0)},
        };

        for (const auto& key : keys) {
            ++stats[0].counts[add_hash(key, m)];
            ++stats[1].counts[fnv1a_hash(key, m)];
        }

        const double expected = static_cast<double>(n) / static_cast<double>(m);

        for (const auto& s : stats) {
            int empty = 0;
            int longest = 0;
            double sum = 0.0;
            double sumsq = 0.0;
            int nonempty = 0;

            for (int c : s.counts) {
                if (c == 0) ++empty;
                if (c > longest) longest = c;
                sum += static_cast<double>(c);
                sumsq += static_cast<double>(c) * static_cast<double>(c);
                if (c > 0) ++nonempty;
            }

            const double mean_all = sum / static_cast<double>(m);
            const double variance = sumsq / static_cast<double>(m) - mean_all * mean_all;
            const double mean_nonempty = (nonempty > 0) ? (sum / static_cast<double>(nonempty)) : 0.0;

            double chi2 = 0.0;
            if (expected > 0.0) {
                for (int c : s.counts) {
                    const double diff = static_cast<double>(c) - expected;
                    chi2 += diff * diff / expected;
                }
            }

            std::cout << s.name << "," << m << "," << n << ","
                      << empty << "," << longest << ","
                      << std::fixed << std::setprecision(2) << mean_nonempty << ","
                      << std::fixed << std::setprecision(2) << variance << ","
                      << std::fixed << std::setprecision(2) << chi2 << "\n";
        }
    }

    return 0;
}
