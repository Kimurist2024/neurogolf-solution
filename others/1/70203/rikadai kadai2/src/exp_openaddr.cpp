#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// FNV-1a 64-bit hash
static std::uint64_t fnv1a(const std::string& key) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : key) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

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

// ---------- chained hashing ----------
class ChainedTable {
    struct Node {
        std::string key;
        Node* next;
    };
    std::vector<Node*> buckets_;

public:
    explicit ChainedTable(std::size_t cap) : buckets_(cap, nullptr) {}

    ~ChainedTable() {
        for (Node* head : buckets_) {
            while (head) {
                Node* n = head;
                head = head->next;
                delete n;
            }
        }
    }

    std::size_t capacity() const { return buckets_.size(); }

    void insert(const std::string& key) {
        std::size_t h = fnv1a(key) & (buckets_.size() - 1);
        Node* n = new Node{key, buckets_[h]};
        buckets_[h] = n;
    }

    // return number of probes (bucket reference + key comparisons)
    int find(const std::string& key) const {
        std::size_t h = fnv1a(key) & (buckets_.size() - 1);
        int probes = 1;  // bucket reference
        for (Node* p = buckets_[h]; p; p = p->next) {
            if (p->key == key) {
                return probes;
            }
            ++probes;  // key comparison that failed
        }
        return probes;
    }
};

// ---------- open addressing with linear probing ----------
class OpenAddrTable {
    enum State : std::uint8_t { EMPTY = 0, USED = 1, DELETED = 2 };
    struct Slot {
        std::string key;
        State state = EMPTY;
    };
    std::vector<Slot> slots_;
    std::size_t mask_;

public:
    explicit OpenAddrTable(std::size_t cap) : slots_(cap), mask_(cap - 1) {}

    std::size_t capacity() const { return slots_.size(); }

    void insert(const std::string& key) {
        std::size_t i = fnv1a(key) & mask_;
        while (slots_[i].state == USED) {
            if (slots_[i].key == key) {
                return;  // already present; not expected in this experiment
            }
            i = (i + 1) & mask_;
        }
        slots_[i].key = key;
        slots_[i].state = USED;
    }

    // successful search: probe count = slots examined until match
    int find_success(const std::string& key) const {
        std::size_t i = fnv1a(key) & mask_;
        int probes = 0;
        for (;;) {
            ++probes;
            if (slots_[i].state == EMPTY) {
                return probes;  // not found, but this path is not used for success
            }
            if (slots_[i].state == USED && slots_[i].key == key) {
                return probes;
            }
            i = (i + 1) & mask_;
        }
    }

    // unsuccessful search: probe count until EMPTY slot is seen
    int find_unsucc(const std::string& key) const {
        std::size_t i = fnv1a(key) & mask_;
        int probes = 0;
        for (;;) {
            ++probes;
            if (slots_[i].state == EMPTY) {
                return probes;
            }
            if (slots_[i].state == USED && slots_[i].key == key) {
                return probes;  // found, not expected for unsucc set
            }
            i = (i + 1) & mask_;
        }
    }
};

int main() {
    constexpr std::size_t CAPACITY = 1ULL << 17;  // 131072
    constexpr int KEY_LEN = 12;

    std::mt19937 rng(42);

    const std::vector<double> alphas = {0.25, 0.5, 0.7, 0.8, 0.9, 0.95};

    std::cout << "method,alpha,n,succ_probes,unsucc_probes,succ_ns,unsucc_ns\n";

    for (double alpha : alphas) {
        std::size_t n = static_cast<std::size_t>(std::llround(alpha * CAPACITY));

        // generate distinct inserted keys and distinct non-inserted keys
        std::vector<std::string> inserted;
        inserted.reserve(n);
        while (inserted.size() < n) {
            std::string k = random_key(rng, KEY_LEN);
            inserted.push_back(std::move(k));
        }

        std::vector<std::string> not_inserted;
        not_inserted.reserve(n);
        while (not_inserted.size() < n) {
            std::string k = random_key(rng, KEY_LEN);
            not_inserted.push_back(std::move(k));
        }

        // ---- chained ----
        {
            ChainedTable ct(CAPACITY);
            for (const auto& k : inserted) {
                ct.insert(k);
            }

            long long succ_found = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            double succ_probes = 0.0;
            for (const auto& k : inserted) {
                int p = ct.find(k);
                succ_probes += p;
                ++succ_found;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double succ_ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / n;

            auto t2 = std::chrono::high_resolution_clock::now();
            double unsucc_probes = 0.0;
            for (const auto& k : not_inserted) {
                unsucc_probes += ct.find(k);
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            double unsucc_ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count()) / n;

            std::cerr << "[chained] alpha=" << alpha
                      << " inserted=" << succ_found
                      << " not_inserted=" << not_inserted.size() << "\n";

            std::cout << "chained," << alpha << "," << n << ","
                      << (succ_probes / n) << "," << (unsucc_probes / n) << ","
                      << succ_ns << "," << unsucc_ns << "\n";
        }

        // ---- open addressing ----
        {
            OpenAddrTable ot(CAPACITY);
            for (const auto& k : inserted) {
                ot.insert(k);
            }

            long long succ_found = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            double succ_probes = 0.0;
            for (const auto& k : inserted) {
                int p = ot.find_success(k);
                succ_probes += p;
                ++succ_found;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double succ_ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / n;

            auto t2 = std::chrono::high_resolution_clock::now();
            double unsucc_probes = 0.0;
            for (const auto& k : not_inserted) {
                unsucc_probes += ot.find_unsucc(k);
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            double unsucc_ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count()) / n;

            std::cerr << "[openaddr] alpha=" << alpha
                      << " inserted=" << succ_found
                      << " not_inserted=" << not_inserted.size() << "\n";

            std::cout << "openaddr," << alpha << "," << n << ","
                      << (succ_probes / n) << "," << (unsucc_probes / n) << ","
                      << succ_ns << "," << unsucc_ns << "\n";
        }
    }

    return 0;
}
