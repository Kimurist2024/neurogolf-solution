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

static int djb2_hash(const std::string& key, int m) {
    unsigned long h = 5381UL;
    for (unsigned char c : key) {
        h = h * 33UL + static_cast<unsigned long>(c);
    }
    return static_cast<int>(h % static_cast<unsigned long>(m));
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

static int std_hash(const std::string& key, int m) {
    std::size_t h = std::hash<std::string>{}(key);
    return static_cast<int>(h % static_cast<std::size_t>(m));
}

struct Stats {
    std::string name;
    std::vector<int> counts;
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " keyfile m\n";
        return 1;
    }

    const std::string keyfile = argv[1];
    const int m = std::atoi(argv[2]);
    if (m <= 0) {
        std::cerr << "m must be positive\n";
        return 1;
    }

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

    std::vector<Stats> stats = {
        {"add",     std::vector<int>(m, 0)},
        {"djb2",    std::vector<int>(m, 0)},
        {"fnv1a",   std::vector<int>(m, 0)},
        {"stdhash", std::vector<int>(m, 0)},
    };

    for (const auto& key : keys) {
        ++stats[0].counts[add_hash(key, m)];
        ++stats[1].counts[djb2_hash(key, m)];
        ++stats[2].counts[fnv1a_hash(key, m)];
        ++stats[3].counts[std_hash(key, m)];
    }

    std::cout << "hash,bucket,count\n";
    for (const auto& s : stats) {
        for (int b = 0; b < m; ++b) {
            std::cout << s.name << "," << b << "," << s.counts[b] << "\n";
        }
    }

    const double n = static_cast<double>(keys.size());
    const double expected = n / static_cast<double>(m);

    std::cerr << "-------------------------------------------------------------\n";
    std::cerr << "Hash collision distribution summary (m=" << m << ", n=" << keys.size() << ")\n";
    std::cerr << "-------------------------------------------------------------\n";
    std::cerr << std::left << std::setw(10) << "function"
              << std::right << std::setw(8) << "empty"
              << std::setw(10) << "longest"
              << std::setw(18) << "avg(non-empty)"
              << std::setw(14) << "variance"
              << std::setw(14) << "chi2" << "\n";
    std::cerr << "-------------------------------------------------------------\n";

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

        std::cerr << std::left << std::setw(10) << s.name
                  << std::right << std::setw(8) << empty
                  << std::setw(10) << longest
                  << std::fixed << std::setprecision(2)
                  << std::setw(18) << mean_nonempty
                  << std::setw(14) << variance
                  << std::setw(14) << chi2 << "\n";
    }
    std::cerr << "-------------------------------------------------------------\n";

    return 0;
}
