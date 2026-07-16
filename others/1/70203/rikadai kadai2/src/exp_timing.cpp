#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================
// fixed17: 加算ハッシュ、バケット数 17 固定のチェイン法
// ============================================================
class Fixed17Table {
public:
    struct Node {
        std::string key;
        int value;
        Node* next;
    };

    Fixed17Table() : size_(0) {
        for (int i = 0; i < 17; ++i) {
            buckets_[i] = nullptr;
        }
    }

    ~Fixed17Table() {
        clear();
    }

    Fixed17Table(const Fixed17Table&) = delete;
    Fixed17Table& operator=(const Fixed17Table&) = delete;

    void insert(const std::string& key, int value) {
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
        for (int i = 0; i < 17; ++i) {
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

private:
    static int hash_(const std::string& key) {
        int h = 0;
        for (unsigned char c : key) {
            h += static_cast<int>(c);
        }
        return h % 17;
    }

    Node* buckets_[17];
    int size_;
};

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
// 共通ラッパー
// ============================================================
struct WrapperBase {
    virtual ~WrapperBase() = default;
    virtual void insert(const std::string& key, int value) = 0;
    virtual bool find(const std::string& key) = 0;
};

template <class Table>
struct GenericWrapper : public WrapperBase {
    Table table;
    void insert(const std::string& key, int value) override {
        table.insert(key, value);
    }
    bool find(const std::string& key) override {
        return table.find(key);
    }
};

struct UnorderedMapWrapper : public WrapperBase {
    std::unordered_map<std::string, int> table;
    void insert(const std::string& key, int value) override {
        table.emplace(key, value);
    }
    bool find(const std::string& key) override {
        return table.find(key) != table.end();
    }
};

struct MapWrapper : public WrapperBase {
    std::map<std::string, int> table;
    void insert(const std::string& key, int value) override {
        table.emplace(key, value);
    }
    bool find(const std::string& key) override {
        return table.find(key) != table.end();
    }
};

struct Case {
    std::string name;
    bool skip_if_too_slow;
    std::unique_ptr<WrapperBase> (*create)();
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

static std::vector<std::string> generate_lookups(
    const std::vector<std::string>& keys, int count) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(keys.size()) - 1);
    std::vector<std::string> lookups;
    lookups.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        lookups.push_back(keys[dist(rng)]);
    }
    return lookups;
}

// ============================================================
// メイン
// ============================================================
int main() {
    const std::vector<int> Ns = {
        1000, 3162, 10000, 31623, 100000, 316228, 1000000
    };
    const int lookup_count = 100000;

    const std::vector<Case> cases = {
        {"fixed17", true, []() -> std::unique_ptr<WrapperBase> {
             return std::make_unique<GenericWrapper<Fixed17Table>>();
         }},
        {"improved", false, []() -> std::unique_ptr<WrapperBase> {
             return std::make_unique<GenericWrapper<ImprovedTable>>();
         }},
        {"unordered_map", false, []() -> std::unique_ptr<WrapperBase> {
             return std::make_unique<UnorderedMapWrapper>();
         }},
        {"map", false, []() -> std::unique_ptr<WrapperBase> {
             return std::make_unique<MapWrapper>();
         }},
    };

    std::cout << "structure,N,rep,insert_total_us,lookup_avg_ns\n";

    using clock = std::chrono::steady_clock;

    for (int N : Ns) {
        const auto keys = generate_unique_keys(N);
        const auto lookups = generate_lookups(keys, lookup_count);

        for (const auto& c : cases) {
            if (c.skip_if_too_slow && N > 100000) {
                std::cerr << "skip " << c.name << " for N=" << N << " (too slow)\n";
                continue;
            }
            for (int rep = 1; rep <= 5; ++rep) {
                std::cerr << "Running " << c.name << " N=" << N << " rep=" << rep << "\n";
                auto table = c.create();

                const auto t0 = clock::now();
                for (int i = 0; i < N; ++i) {
                    table->insert(keys[i], i);
                }
                const auto t1 = clock::now();

                long long found = 0;
                const auto t2 = clock::now();
                for (const auto& k : lookups) {
                    found += table->find(k) ? 1 : 0;
                }
                const auto t3 = clock::now();
                std::cerr << "  found=" << found << "/" << lookup_count << "\n";

                const double insert_us = static_cast<double>(
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
                const double lookup_total_ns = static_cast<double>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
                const double lookup_avg_ns = lookup_total_ns / static_cast<double>(lookup_count);

                std::cout << c.name << "," << N << "," << rep << ","
                          << std::fixed << std::setprecision(1) << insert_us << ","
                          << std::fixed << std::setprecision(1) << lookup_avg_ns << "\n";
            }
        }
    }

    return 0;
}
