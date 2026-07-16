#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

// ============================================================
// fixed17: 加算ハッシュ、バケット数 17 固定のチェイン法
// 検索時のキー比較回数を返せる
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

    // 発見したら true。比較回数は comparisons に書き込む。
    bool find(const std::string& key, long long& comparisons) const {
        const int h = hash_(key);
        comparisons = 0;
        for (Node* p = buckets_[h]; p != nullptr; p = p->next) {
            ++comparisons;
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

    int size() const {
        return size_;
    }

    int max_chain() const {
        int max_len = 0;
        for (int i = 0; i < 17; ++i) {
            int len = 0;
            for (Node* p = buckets_[i]; p != nullptr; p = p->next) {
                ++len;
            }
            if (len > max_len) {
                max_len = len;
            }
        }
        return max_len;
    }

    int used_buckets() const {
        int count = 0;
        for (int i = 0; i < 17; ++i) {
            if (buckets_[i] != nullptr) {
                ++count;
            }
        }
        return count;
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
// improved: FNV-1a 32bit + 負荷率 1.0 超で素数リハッシュ
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

    bool find(const std::string& key, long long& comparisons) const {
        const int h = hash_(key);
        comparisons = 0;
        for (Node* p = buckets_[h]; p != nullptr; p = p->next) {
            ++comparisons;
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

    int size() const {
        return size_;
    }

    int bucket_count() const {
        return bucket_count_;
    }

    int max_chain() const {
        int max_len = 0;
        for (int i = 0; i < bucket_count_; ++i) {
            int len = 0;
            for (Node* p = buckets_[i]; p != nullptr; p = p->next) {
                ++len;
            }
            if (len > max_len) {
                max_len = len;
            }
        }
        return max_len;
    }

    int used_buckets() const {
        int count = 0;
        for (int i = 0; i < bucket_count_; ++i) {
            if (buckets_[i] != nullptr) {
                ++count;
            }
        }
        return count;
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
        return static_cast<int>(
            fnv1a_(key) % static_cast<std::uint32_t>(bucket_count_));
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
static int additive_hash_mod17(const std::string& key) {
    int h = 0;
    for (unsigned char c : key) {
        h += static_cast<int>(c);
    }
    return h % 17;
}

static std::vector<std::string> generate_attack_keys(int N) {
    std::string base = "aaabbbcccddd";
    std::sort(base.begin(), base.end());

    std::vector<std::string> keys;
    keys.reserve(static_cast<std::size_t>(N));

    do {
        keys.push_back(base);
        if (static_cast<int>(keys.size()) >= N) {
            break;
        }
    } while (std::next_permutation(base.begin(), base.end()));

    if (static_cast<int>(keys.size()) < N) {
        std::cerr << "warning: could not generate enough attack keys ("
                  << keys.size() << " / " << N << ")\n";
    }
    return keys;
}

static std::vector<std::string> generate_random_keys(int N) {
    static const char chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    static const int L = 62;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, L - 1);

    std::unordered_set<std::string> seen;
    seen.reserve(static_cast<std::size_t>(N) * 2);

    std::vector<std::string> keys;
    keys.reserve(static_cast<std::size_t>(N));

    std::string key(12, ' ');
    while (static_cast<int>(keys.size()) < N) {
        for (int i = 0; i < 12; ++i) {
            key[i] = chars[dist(rng)];
        }
        if (seen.insert(key).second) {
            keys.push_back(key);
        }
    }
    return keys;
}

// ============================================================
// 計測
// ============================================================
struct MeasureResult {
    double avg_comparisons;
    double avg_lookup_ns;
};

template <class Table>
static MeasureResult measure(Table& table, const std::vector<std::string>& keys) {
    const int n = static_cast<int>(keys.size());
    const int lookup_repeats = 5;

    // 挿入
    for (int i = 0; i < n; ++i) {
        table.insert(keys[i], i);
    }

    // 決定的な平均キー比較回数
    long long total_comparisons = 0;
    long long found_count = 0;
    for (int r = 0; r < lookup_repeats; ++r) {
        for (int i = 0; i < n; ++i) {
            long long comp = 0;
            const bool found = table.find(keys[i], comp);
            if (found) {
                ++found_count;
            }
            total_comparisons += comp;
        }
    }
    const double avg_comparisons =
        static_cast<double>(total_comparisons) /
        static_cast<double>(n * lookup_repeats);

    if (found_count != static_cast<long long>(n) * lookup_repeats) {
        std::cerr << "error: some keys were not found during measurement\n";
    }

    // 平均検索時間: 9回反復の最小値
    // コンパイラの最適化で検索ループが消えないよう、volatile sink に結果を加算
    using clock = std::chrono::steady_clock;
    long long best_total_ns = -1;
    const int time_repeats = 9;
    volatile long long time_sink = 0;
    for (int iter = 0; iter < time_repeats; ++iter) {
        const auto t0 = clock::now();
        for (int r = 0; r < lookup_repeats; ++r) {
            for (int i = 0; i < n; ++i) {
                long long comp = 0;
                table.find(keys[i], comp);
                time_sink += comp;
            }
        }
        const auto t1 = clock::now();
        const long long ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        if (best_total_ns < 0 || ns < best_total_ns) {
            best_total_ns = ns;
        }
    }
    if (time_sink < 0) {
        std::cerr << time_sink << "\n";
    }
    const double avg_lookup_ns =
        static_cast<double>(best_total_ns) /
        static_cast<double>(n * lookup_repeats);

    return {avg_comparisons, avg_lookup_ns};
}

template <class Table>
static void report_stats(const std::string& name, const Table& table) {
    std::cerr << name << ": "
              << "size=" << table.size()
              << " buckets=" << table.used_buckets()
              << " max_chain=" << table.max_chain()
              << " load_factor=" << std::fixed << std::setprecision(3)
              << (static_cast<double>(table.size()) /
                  static_cast<double>(table.used_buckets()))
              << "\n";
}

// ============================================================
// メイン
// ============================================================
int main() {
    const int N = 2000;

    const auto attack_keys = generate_attack_keys(N);
    const auto random_keys = generate_random_keys(N);

    // attack キーの加算ハッシュが全て同一か std::set で検証
    std::set<int> attack_hashes;
    for (const auto& k : attack_keys) {
        attack_hashes.insert(additive_hash_mod17(k));
    }
    std::cerr << "attack keys additive hash (mod 17):";
    for (int h : attack_hashes) {
        std::cerr << " " << h;
    }
    std::cerr << (attack_hashes.size() == 1 ? " (all same)\n" : " (DIFFERENT!)\n");

    std::cerr << "attack keys: " << attack_keys.size()
              << ", random keys: " << random_keys.size() << "\n";

    // (a) fixed17 / attack
    Fixed17Table fixed17_attack;
    const auto r1 = measure(fixed17_attack, attack_keys);
    report_stats("fixed17/attack", fixed17_attack);

    // (b) fixed17 / random
    Fixed17Table fixed17_random;
    const auto r2 = measure(fixed17_random, random_keys);
    report_stats("fixed17/random", fixed17_random);

    // (c) improved / attack
    ImprovedTable improved_attack;
    const auto r3 = measure(improved_attack, attack_keys);
    report_stats("improved/attack", improved_attack);

    // (d) improved / random
    ImprovedTable improved_random;
    const auto r4 = measure(improved_random, random_keys);
    report_stats("improved/random", improved_random);

    std::cout << "table,keyset,n,avg_comparisons,avg_lookup_ns\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "fixed17,attack," << N << "," << r1.avg_comparisons << ","
              << r1.avg_lookup_ns << "\n";
    std::cout << "fixed17,random," << N << "," << r2.avg_comparisons << ","
              << r2.avg_lookup_ns << "\n";
    std::cout << "improved,attack," << N << "," << r3.avg_comparisons << ","
              << r3.avg_lookup_ns << "\n";
    std::cout << "improved,random," << N << "," << r4.avg_comparisons << ","
              << r4.avg_lookup_ns << "\n";

    return 0;
}
