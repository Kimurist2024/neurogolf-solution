#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ---- (a) improved 相当：タグなし、FNV-1a + 素数リハッシュのチェイン法 ----
template <class T>
class HashTableBaseline {
public:
    struct Node {
        std::string key;
        T value;
        Node* next;
    };

    HashTableBaseline()
        : buckets_(nullptr), bucket_count_(17), size_(0) {
        buckets_ = new Node*[bucket_count_];
        for (int i = 0; i < bucket_count_; ++i) {
            buckets_[i] = nullptr;
        }
    }

    ~HashTableBaseline() {
        for (int i = 0; i < bucket_count_; ++i) {
            Node* p = buckets_[i];
            while (p != nullptr) {
                Node* next = p->next;
                delete p;
                p = next;
            }
        }
        delete[] buckets_;
    }

    HashTableBaseline(const HashTableBaseline&) = delete;
    HashTableBaseline& operator=(const HashTableBaseline&) = delete;

    static std::uint32_t fnv1a(const std::string& key) {
        std::uint32_t hashval = 2166136261u;
        for (unsigned char c : key) {
            hashval ^= c;
            hashval *= 16777619u;
        }
        return hashval;
    }

    int hash(const std::string& key) const {
        return static_cast<int>(fnv1a(key) % static_cast<std::uint32_t>(bucket_count_));
    }

    static bool is_prime(int n) {
        if (n < 2) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        for (int i = 3; i * i <= n; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    static int next_prime(int current) {
        int candidate = current * 2 + 1;
        while (!is_prime(candidate)) {
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
                int h = static_cast<int>(fnv1a(p->key) % static_cast<std::uint32_t>(new_bucket_count));
                p->next = new_buckets[h];
                new_buckets[h] = p;
                p = next;
            }
        }
        delete[] buckets_;
        buckets_ = new_buckets;
        bucket_count_ = new_bucket_count;
    }

    bool insert(const std::string& key, const T& value) {
        if (load_factor() > 1.0) {
            rehash(next_prime(bucket_count_));
        }
        int h = hash(key);
        Node* p = new Node{key, value, buckets_[h]};
        buckets_[h] = p;
        ++size_;
        return true;
    }

    const T* operator()(const std::string& key) const {
        int h = hash(key);
        Node* p = buckets_[h];
        while (p != nullptr) {
            ++strcmp_count_;
            if (p->key == key) {
                return &(p->value);
            }
            p = p->next;
        }
        return nullptr;
    }

    int size() const { return size_; }
    int bucket_count() const { return bucket_count_; }
    double load_factor() const {
        return static_cast<double>(size_) / static_cast<double>(bucket_count_);
    }
    void reset_counters() const { strcmp_count_ = 0; }
    long long strcmp_count() const { return strcmp_count_; }

private:
    Node** buckets_;
    int bucket_count_;
    int size_;
    mutable long long strcmp_count_ = 0;
};

// ---- (b) tagged：タグ付き比較 + リハッシュ時は保存済みハッシュを利用 ----
template <class T>
class HashTableTagged {
public:
    struct Node {
        std::string key;
        T value;
        std::uint32_t hash;
        Node* next;
    };

    HashTableTagged()
        : buckets_(nullptr), bucket_count_(17), size_(0) {
        buckets_ = new Node*[bucket_count_];
        for (int i = 0; i < bucket_count_; ++i) {
            buckets_[i] = nullptr;
        }
    }

    ~HashTableTagged() {
        for (int i = 0; i < bucket_count_; ++i) {
            Node* p = buckets_[i];
            while (p != nullptr) {
                Node* next = p->next;
                delete p;
                p = next;
            }
        }
        delete[] buckets_;
    }

    HashTableTagged(const HashTableTagged&) = delete;
    HashTableTagged& operator=(const HashTableTagged&) = delete;

    static std::uint32_t fnv1a(const std::string& key) {
        std::uint32_t hashval = 2166136261u;
        for (unsigned char c : key) {
            hashval ^= c;
            hashval *= 16777619u;
        }
        return hashval;
    }

    static bool is_prime(int n) {
        if (n < 2) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        for (int i = 3; i * i <= n; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    static int next_prime(int current) {
        int candidate = current * 2 + 1;
        while (!is_prime(candidate)) {
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
                int h = static_cast<int>(p->hash % static_cast<std::uint32_t>(new_bucket_count));
                p->next = new_buckets[h];
                new_buckets[h] = p;
                p = next;
            }
        }
        delete[] buckets_;
        buckets_ = new_buckets;
        bucket_count_ = new_bucket_count;
    }

    bool insert(const std::string& key, const T& value) {
        if (load_factor() > 1.0) {
            rehash(next_prime(bucket_count_));
        }
        std::uint32_t hval = fnv1a(key);
        int h = static_cast<int>(hval % static_cast<std::uint32_t>(bucket_count_));
        Node* p = new Node{key, value, hval, buckets_[h]};
        buckets_[h] = p;
        ++size_;
        return true;
    }

    const T* operator()(const std::string& key) const {
        std::uint32_t hval = fnv1a(key);
        int h = static_cast<int>(hval % static_cast<std::uint32_t>(bucket_count_));
        Node* p = buckets_[h];
        while (p != nullptr) {
            if (p->hash == hval) {
                ++strcmp_count_;
                if (p->key == key) {
                    return &(p->value);
                }
            }
            p = p->next;
        }
        return nullptr;
    }

    int size() const { return size_; }
    int bucket_count() const { return bucket_count_; }
    double load_factor() const {
        return static_cast<double>(size_) / static_cast<double>(bucket_count_);
    }
    void reset_counters() const { strcmp_count_ = 0; }
    long long strcmp_count() const { return strcmp_count_; }

private:
    Node** buckets_;
    int bucket_count_;
    int size_;
    mutable long long strcmp_count_ = 0;
};

// ---- ユーティリティ ----
static std::string make_key(std::mt19937& gen, int len) {
    static const char chars[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<int> dist(0, 61);
    std::string s;
    s.reserve(len);
    for (int i = 0; i < len; ++i) {
        s += chars[dist(gen)];
    }
    return s;
}

template <class Table>
static void measure(const std::string& name, int keylen,
                    const std::vector<std::string>& keys,
                    const std::vector<std::string>& missing_keys,
                    long long& out_insert_us,
                    long long& out_succ_ns,
                    long long& out_unsucc_ns,
                    double& out_succ_strcmp,
                    double& out_unsucc_strcmp) {
    using namespace std::chrono;
    constexpr int iters = 9;
    constexpr int lookups = 100000;

    out_insert_us = std::numeric_limits<long long>::max();
    out_succ_ns = std::numeric_limits<long long>::max();
    out_unsucc_ns = std::numeric_limits<long long>::max();

    std::mt19937 gen(42);
    std::uniform_int_distribution<std::size_t> idx_dist(0, keys.size() - 1);

    for (int iter = 0; iter < iters; ++iter) {
        Table table;

        // (1) 挿入時間
        auto t0 = steady_clock::now();
        for (const auto& k : keys) {
            if constexpr (std::is_same_v<Table, std::unordered_map<std::string, int>>) {
                table.emplace(k, 1);
            } else {
                table.insert(k, 1);
            }
        }
        auto t1 = steady_clock::now();
        long long insert_us = duration_cast<microseconds>(t1 - t0).count();
        out_insert_us = std::min(out_insert_us, insert_us);

        auto do_lookup = [](const Table& tbl, const std::string& k) -> const int* {
            if constexpr (std::is_same_v<Table, std::unordered_map<std::string, int>>) {
                auto it = tbl.find(k);
                return (it != tbl.end()) ? &(it->second) : nullptr;
            } else {
                return tbl(k);
            }
        };

        // (2) 成功検索時間
        long long found = 0;
        t0 = steady_clock::now();
        for (int i = 0; i < lookups; ++i) {
            const int* v = do_lookup(table, keys[idx_dist(gen)]);
            if (v != nullptr) {
                ++found;
            }
        }
        t1 = steady_clock::now();
        long long succ_total_ns = duration_cast<nanoseconds>(t1 - t0).count();
        out_succ_ns = std::min(out_succ_ns, succ_total_ns / lookups);
        std::cerr << name << " L=" << keylen << " iter=" << iter
                  << " found=" << found << std::endl;

        // (3) 不成功検索時間
        found = 0;
        t0 = steady_clock::now();
        for (int i = 0; i < lookups; ++i) {
            const int* v = do_lookup(table, missing_keys[i]);
            if (v != nullptr) {
                ++found;
            }
        }
        t1 = steady_clock::now();
        long long unsucc_total_ns = duration_cast<nanoseconds>(t1 - t0).count();
        out_unsucc_ns = std::min(out_unsucc_ns, unsucc_total_ns / lookups);
        std::cerr << name << " L=" << keylen << " iter=" << iter
                  << " unsucc_found=" << found << std::endl;
    }

    // (4) 決定的な指標: 検索 1 回あたりの文字列比較回数(時間と違い環境に依存しない)
    out_succ_strcmp = -1.0;
    out_unsucc_strcmp = -1.0;
    if constexpr (!std::is_same_v<Table, std::unordered_map<std::string, int>>) {
        Table table;
        for (const auto& k : keys) {
            table.insert(k, 1);
        }
        std::mt19937 g2(42);
        std::uniform_int_distribution<std::size_t> d2(0, keys.size() - 1);
        table.reset_counters();
        for (int i = 0; i < lookups; ++i) {
            (void)table(keys[d2(g2)]);
        }
        out_succ_strcmp = static_cast<double>(table.strcmp_count()) / lookups;
        table.reset_counters();
        for (int i = 0; i < lookups; ++i) {
            (void)table(missing_keys[i]);
        }
        out_unsucc_strcmp = static_cast<double>(table.strcmp_count()) / lookups;
    }
}

template <class Table>
static void run_for_table(const std::string& name,
                          const std::vector<std::vector<std::string>>& all_keys,
                          const std::vector<std::vector<std::string>>& all_missing) {
    const int lens[] = {12, 64, 256};
    for (std::size_t i = 0; i < all_keys.size(); ++i) {
        long long insert_us, succ_ns, unsucc_ns;
        double succ_strcmp, unsucc_strcmp;
        measure<Table>(name, lens[i], all_keys[i], all_missing[i],
                       insert_us, succ_ns, unsucc_ns, succ_strcmp, unsucc_strcmp);
        std::cout << name << "," << lens[i] << ","
                  << insert_us << "," << succ_ns << "," << unsucc_ns << ","
                  << std::fixed << std::setprecision(5) << succ_strcmp << ","
                  << unsucc_strcmp << std::endl;
    }
}

int main() {
    constexpr int N = 200000;
    constexpr int lookups = 100000;
    const int lens[] = {12, 64, 256};

    std::mt19937 gen_insert(42);
    std::mt19937 gen_missing(123456789);

    std::vector<std::vector<std::string>> all_keys(3);
    std::vector<std::vector<std::string>> all_missing(3);
    for (int i = 0; i < 3; ++i) {
        int L = lens[i];
        all_keys[i].reserve(N);
        all_missing[i].reserve(lookups);
        for (int j = 0; j < N; ++j) {
            all_keys[i].push_back(make_key(gen_insert, L));
        }
        for (int j = 0; j < lookups; ++j) {
            all_missing[i].push_back(make_key(gen_missing, L));
        }
    }

    std::cout << "structure,keylen,insert_total_us,succ_lookup_ns,unsucc_lookup_ns,succ_strcmp,unsucc_strcmp" << std::endl;
    run_for_table<HashTableBaseline<int>>("baseline", all_keys, all_missing);
    run_for_table<HashTableTagged<int>>("tagged", all_keys, all_missing);
    run_for_table<std::unordered_map<std::string, int>>("unordered_map", all_keys, all_missing);

    return 0;
}
