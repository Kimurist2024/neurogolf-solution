#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static constexpr std::size_t N = 10000;
static constexpr std::size_t BUCKET_COUNT = 16384;  // 2^14, load factor < 1
static constexpr std::size_t BUCKET_MASK = BUCKET_COUNT - 1;

static std::string random_key(std::mt19937& rng, int len) {
    static const char alnum[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789";
    std::uniform_int_distribution<int> dist(0, 61);
    std::string s;
    s.reserve(len);
    for (int i = 0; i < len; ++i) {
        s.push_back(alnum[dist(rng)]);
    }
    return s;
}

// FNV-1a 64-bit hash
static std::uint64_t fnv1a(const std::string& key) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : key) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

// additive hash (same style as hashtable.cpp but with a larger bucket count)
static int additive_hash(const std::string& key) {
    int sum = 0;
    for (unsigned char c : key) {
        sum += c;
    }
    return sum % static_cast<int>(BUCKET_COUNT);
}

// chained hash table using FNV-1a + bucket mask
class ChainedTable {
    struct Node {
        std::string key;
        int value;
        Node* next;
    };
    std::vector<Node*> buckets_;

public:
    ChainedTable() : buckets_(BUCKET_COUNT, nullptr) {}
    ~ChainedTable() {
        for (Node* head : buckets_) {
            while (head) {
                Node* n = head;
                head = head->next;
                delete n;
            }
        }
    }

    void insert(const std::string& key, int value) {
        std::size_t h = fnv1a(key) & BUCKET_MASK;
        Node* n = new Node{key, value, buckets_[h]};
        buckets_[h] = n;
    }

    const int* find(const std::string& key) const {
        std::size_t h = fnv1a(key) & BUCKET_MASK;
        for (Node* p = buckets_[h]; p; p = p->next) {
            if (p->key == key) {
                return &(p->value);
            }
        }
        return nullptr;
    }
};

// OS 由来のノイズを抑えるため反復測定の最小値を採る(マイクロベンチマークの慣行)
static double min_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[0];
}

int main() {
    std::mt19937 rng(42);
    const std::vector<int> lengths = {8, 16, 32, 64, 128, 256};
    const int reps = 15;

    std::cout << "keylen,fnv1a_hash_ns,add_hash_ns,lookup_ns\n";

    for (int len : lengths) {
        std::vector<std::string> keys;
        keys.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            keys.push_back(random_key(rng, len));
        }

        std::uint64_t fnv_total = 0;
        std::uint64_t add_total = 0;
        long long found = 0;
        std::vector<double> fnv_samples, add_samples, lookup_samples;

        ChainedTable table;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            table.insert(keys[i], static_cast<int>(i));
        }

        for (int r = 0; r < reps; ++r) {
            // (1) FNV-1a hash only
            auto t0 = std::chrono::steady_clock::now();
            for (const auto& k : keys) {
                fnv_total += fnv1a(k);
            }
            auto t1 = std::chrono::steady_clock::now();
            fnv_samples.push_back(static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / N);

            // (2) additive hash only
            auto t2 = std::chrono::steady_clock::now();
            for (const auto& k : keys) {
                add_total += static_cast<std::uint64_t>(additive_hash(k));
            }
            auto t3 = std::chrono::steady_clock::now();
            add_samples.push_back(static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count()) / N);

            // (3) look up all keys in the chained table
            auto t4 = std::chrono::steady_clock::now();
            for (const auto& k : keys) {
                const int* p = table.find(k);
                if (p != nullptr) {
                    ++found;
                }
            }
            auto t5 = std::chrono::steady_clock::now();
            lookup_samples.push_back(static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count()) / N);
        }

        std::cerr << "[keylen=" << len << "] found=" << found
                  << " fnv_total=" << fnv_total
                  << " add_total=" << add_total << "\n";

        std::cout << len << "," << min_of(fnv_samples) << ","
                  << min_of(add_samples) << "," << min_of(lookup_samples) << "\n";
    }

    return 0;
}
