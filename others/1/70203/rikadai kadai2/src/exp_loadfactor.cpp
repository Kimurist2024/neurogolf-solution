#include <cstdint>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static constexpr int M = 17;

struct AddHash {
    static int hash(const std::string& key) {
        int h = 0;
        for (unsigned char c : key) {
            h += static_cast<int>(c);
        }
        return h % M;
    }
};

struct FnvHash {
    static int hash(const std::string& key) {
        std::uint32_t h = 2166136261u;
        for (unsigned char c : key) {
            h ^= static_cast<std::uint32_t>(c);
            h *= 16777619u;
        }
        return static_cast<int>(h % static_cast<std::uint32_t>(M));
    }
};

template <typename Hasher>
class ChainTable {
public:
    struct Node {
        std::string key;
        int value;
        Node* next;
    };

    ChainTable() : size_(0) {
        for (int i = 0; i < M; ++i) {
            buckets_[i] = nullptr;
        }
    }

    ~ChainTable() {
        clear();
    }

    ChainTable(const ChainTable&) = delete;
    ChainTable& operator=(const ChainTable&) = delete;

    void insert(const std::string& key, int value) {
        const int h = Hasher::hash(key);
        buckets_[h] = new Node{key, value, buckets_[h]};
        ++size_;
    }

    // 探索に必要だったキー比較回数を返す。found に成否を書き込む。
    int search_comparisons(const std::string& key, bool& found) const {
        const int h = Hasher::hash(key);
        int cmp = 0;
        for (Node* p = buckets_[h]; p != nullptr; p = p->next) {
            ++cmp;
            if (p->key == key) {
                found = true;
                return cmp;
            }
        }
        found = false;
        return cmp;
    }

    void clear() {
        for (int i = 0; i < M; ++i) {
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
    Node* buckets_[M];
    int size_;
};

template <typename Hasher>
static void run_experiment(const std::vector<std::string>& keys, const std::string& name) {
    const std::vector<int> ns = {8, 17, 34, 51, 85, 170, 340, 510, 850, 1700};

    for (int n : ns) {
        if (n > static_cast<int>(keys.size())) {
            continue;
        }

        ChainTable<Hasher> table;
        for (int i = 0; i < n; ++i) {
            table.insert(keys[i], i);
        }

        long long succ_sum = 0;
        for (int i = 0; i < n; ++i) {
            bool found = false;
            succ_sum += table.search_comparisons(keys[i], found);
        }
        const double succ_avg = static_cast<double>(succ_sum) / static_cast<double>(n);

        long long unsucc_sum = 0;
        // 不変条件: 不成功検索用のキー(末尾 n 個)が挿入済みキー(先頭 n 個)と
        // 重ならないよう n <= keys.size()/2 を要求する
        if (static_cast<std::size_t>(n) > keys.size() / 2) {
            std::cerr << "error: n=" << n << " exceeds half of key file; "
                      << "unsuccessful-search keys would overlap inserted keys\n";
            std::exit(1);
        }
        const std::size_t start = keys.size() - static_cast<std::size_t>(n);
        for (std::size_t i = start; i < keys.size(); ++i) {
            bool found = false;
            unsucc_sum += table.search_comparisons(keys[i], found);
        }
        const double unsucc_avg = static_cast<double>(unsucc_sum) / static_cast<double>(n);
        const double alpha = static_cast<double>(n) / static_cast<double>(M);

        std::cout << name << "," << n << ","
                  << std::fixed << std::setprecision(6) << alpha << ","
                  << std::fixed << std::setprecision(6) << succ_avg << ","
                  << std::fixed << std::setprecision(6) << unsucc_avg << "\n";
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " keyfile\n";
        return 1;
    }

    std::ifstream ifs(argv[1]);
    if (!ifs) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 1;
    }

    std::vector<std::string> keys;
    std::string line;
    while (std::getline(ifs, line)) {
        keys.push_back(line);
    }

    std::cout << "hash,n,alpha,succ_avg,unsucc_avg\n";
    run_experiment<AddHash>(keys, "add");
    run_experiment<FnvHash>(keys, "fnv1a");

    return 0;
}
