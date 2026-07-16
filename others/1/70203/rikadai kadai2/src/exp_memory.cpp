#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <sys/resource.h>
#include <unordered_map>
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

    static std::size_t node_size() {
        return sizeof(Node);
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
    std::vector<std::string> keys;
    keys.reserve(static_cast<std::size_t>(N));

    while (static_cast<int>(keys.size()) < N) {
        keys.push_back(generate_key(rng));
    }
    return keys;
}

// ============================================================
// RSS 取得（macOS では ru_maxrss がバイト単位）
// ============================================================
static long long get_max_rss_bytes() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        std::cerr << "getrusage failed: " << std::strerror(errno) << "\n";
        std::exit(1);
    }
    return static_cast<long long>(usage.ru_maxrss);
}

// ============================================================
// メイン
// ============================================================
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <structure> <N>\n";
        std::cerr << "  structure: improved | unordered_map | map\n";
        return 1;
    }

    const std::string structure = argv[1];
    const int N = std::atoi(argv[2]);
    if (N <= 0) {
        std::cerr << "N must be positive\n";
        return 1;
    }

    std::cerr << "sizeof(std::string)=" << sizeof(std::string)
              << ", sizeof(ImprovedTable::Node)=" << ImprovedTable::node_size() << "\n";

    // キーを先にすべて生成してから RSS を測定
    std::cerr << "Generating " << N << " keys...\n";
    const auto keys = generate_unique_keys(N);

    const long long rss_before = get_max_rss_bytes();
    std::cerr << "rss_before=" << rss_before << " bytes\n";

    if (structure == "improved") {
        ImprovedTable table;
        for (int i = 0; i < N; ++i) {
            table.insert(keys[i], i);
        }
    } else if (structure == "unordered_map") {
        std::unordered_map<std::string, int> table;
        table.reserve(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) {
            table.emplace(keys[i], i);
        }
    } else if (structure == "map") {
        std::map<std::string, int> table;
        for (int i = 0; i < N; ++i) {
            table.emplace(keys[i], i);
        }
    } else {
        std::cerr << "unknown structure: " << structure << "\n";
        return 1;
    }

    const long long rss_after = get_max_rss_bytes();
    const long long delta = rss_after - rss_before;

    std::cout << structure << "," << N << ","
              << rss_before << "," << rss_after << "," << delta << "\n";

    return 0;
}
