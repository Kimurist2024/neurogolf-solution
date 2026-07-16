#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

// ============================================================
// improved: FNV-1a + 負荷率 1.0 超で素数リハッシュ
// ============================================================
class ImprovedTable {
public:
    struct Node {
        std::string key;
        int value;
        Node* next;
    };

    ImprovedTable() : buckets_(nullptr), bucket_count_(17), size_(0) {
        buckets_ = new Node*[bucket_count_];
        for (int i = 0; i < bucket_count_; ++i) {
            buckets_[i] = nullptr;
        }
    }

    ~ImprovedTable() {
        clear();
        delete[] buckets_;
    }

    ImprovedTable(const ImprovedTable&) = delete;
    ImprovedTable& operator=(const ImprovedTable&) = delete;

    void insert(const std::string& key, int value) {
        if (load_factor() > 1.0) {
            rehash(next_prime_(bucket_count_));
        }
        const int h = hash_(key);
        buckets_[h] = new Node{key, value, buckets_[h]};
        ++size_;
    }

    bool find(const std::string& key) const {
        const int h = hash_(key);
        for (Node* p = buckets_[h]; p != nullptr; p = p->next) {
            if (p->key == key) {
                return true;
            }
        }
        return false;
    }

    void clear() {
        for (int i = 0; i < bucket_count_; ++i) {
            Node* p = buckets_[i];
            while (p != nullptr) {
                Node* next = p->next;
                delete p;
                p = next;
            }
            buckets_[i] = nullptr;
        }
        size_ = 0;
    }

    int bucket_count() const {
        return bucket_count_;
    }

    int size() const {
        return size_;
    }

private:
    static std::uint32_t fnv1a_(const std::string& key) {
        std::uint32_t h = 2166136261u;
        for (unsigned char c : key) {
            h ^= static_cast<std::uint32_t>(c);
            h *= 16777619u;
        }
        return h;
    }

    int hash_(const std::string& key) const {
        return static_cast<int>(fnv1a_(key) % static_cast<std::uint32_t>(bucket_count_));
    }

    static bool is_prime_(int n) {
        if (n < 2) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        for (int i = 3; i * i <= n; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    static int next_prime_(int current) {
        int candidate = current * 2 + 1;
        while (!is_prime_(candidate)) {
            candidate += 2;
        }
        return candidate;
    }

    void rehash(int new_bucket_count) {
        Node** new_buckets = new Node*[new_bucket_count];
        for (int i = 0; i < new_bucket_count; ++i) {
            new_buckets[i] = nullptr;
        }
        for (int i = 0; i < bucket_count_; ++i) {
            Node* p = buckets_[i];
            while (p != nullptr) {
                Node* next = p->next;
                const int h = static_cast<int>(
                    fnv1a_(p->key) % static_cast<std::uint32_t>(new_bucket_count));
                p->next = new_buckets[h];
                new_buckets[h] = p;
                p = next;
            }
        }
        delete[] buckets_;
        buckets_ = new_buckets;
        bucket_count_ = new_bucket_count;
    }

    double load_factor() const {
        return static_cast<double>(size_) / static_cast<double>(bucket_count_);
    }

    Node** buckets_;
    int bucket_count_;
    int size_;
};

// ============================================================
// キー生成
// ============================================================
static std::string generate_key(std::mt19937& rng) {
    static const char chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    static const int L = 62;
    std::uniform_int_distribution<int> dist(0, L - 1);

    std::string key(12, ' ');
    for (int i = 0; i < 12; ++i) {
        key[i] = chars[dist(rng)];
    }
    return key;
}

static std::vector<std::string> generate_unique_keys(int N) {
    std::mt19937 rng(42);
    std::unordered_set<std::string> seen;
    seen.reserve(static_cast<std::size_t>(N) * 2);
    std::vector<std::string> keys;
    keys.reserve(static_cast<std::size_t>(N));

    while (static_cast<int>(keys.size()) < N) {
        const std::string k = generate_key(rng);
        if (seen.insert(k).second) {
            keys.push_back(k);
        }
    }
    return keys;
}

// ============================================================
// メイン
// ============================================================
int main() {
    const int N = 200000;
    const long long spike_threshold_ns = 50000;

    std::cerr << "Generating " << N << " unique keys...\n";
    const auto keys = generate_unique_keys(N);

    ImprovedTable table;
    using clock = std::chrono::steady_clock;

    std::cout << "kind,i,value_ns\n";

    long long total_ns = 0;
    long long max_ns = 0;
    int rehash_count = 0;
    int prev_bucket_count = table.bucket_count();

    for (int i = 0; i < N; ++i) {
        if (i % 10000 == 0) {
            std::cerr << "progress: " << i << "/" << N << "\n";
        }

        const auto t0 = clock::now();
        table.insert(keys[i], i);
        const auto t1 = clock::now();

        const long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            t1 - t0).count();
        total_ns += ns;
        if (ns > max_ns) {
            max_ns = ns;
        }

        if (table.bucket_count() != prev_bucket_count) {
            std::cout << "rehash," << i << "," << table.bucket_count() << "\n";
            prev_bucket_count = table.bucket_count();
            ++rehash_count;
        }

        if (i % 100 == 0) {
            std::cout << "sample," << i << "," << ns << "\n";
        }

        if (ns > spike_threshold_ns) {
            std::cout << "spike," << i << "," << ns << "\n";
        }

        if (i % 1000 == 0) {
            const long long cumavg = total_ns / static_cast<long long>(i + 1);
            std::cout << "cumavg," << i << "," << cumavg << "\n";
        }
    }

    const double avg_ns = static_cast<double>(total_ns) / static_cast<double>(N);

    std::cerr << "------------------------------------------------------------\n";
    std::cerr << "Final stats (N=" << N << ")\n";
    std::cerr << "  total insert time: " << total_ns << " ns\n";
    std::cerr << "  average latency:   " << avg_ns << " ns\n";
    std::cerr << "  max latency:       " << max_ns << " ns\n";
    std::cerr << "  rehash count:      " << rehash_count << "\n";
    std::cerr << "  final bucket_count:" << table.bucket_count() << "\n";
    std::cerr << "  final load_factor: "
              << static_cast<double>(table.size()) / static_cast<double>(table.bucket_count())
              << "\n";
    std::cerr << "------------------------------------------------------------\n";

    return 0;
}
