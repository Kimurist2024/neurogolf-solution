#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

// 動的バケット拡張、FNV-1a ハッシュ、およびタグ付き比較を用いた改良版ハッシュ表
template <class T>
class HashTable {
public:
    // チェインのノード（ハッシュ値全体を保存）
    struct Node {
        std::string key;
        T value;
        std::uint32_t hash;
        Node* next;
    };

    // 初期バケット数で空のハッシュ表を作成
    HashTable() : buckets_(nullptr), bucket_count_(17), size_(0),
                  tag_comparisons_(0), string_comparisons_(0) {
        buckets_ = new Node*[bucket_count_];
        for (int i = 0; i < bucket_count_; ++i) {
            buckets_[i] = nullptr;
        }
    }

    // 全ノードとバケット配列を削除
    ~HashTable() {
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

    // コピー禁止
    HashTable(const HashTable&) = delete;
    HashTable& operator=(const HashTable&) = delete;

    // カウンタをリセット
    void reset_counters() const {
        tag_comparisons_ = 0;
        string_comparisons_ = 0;
    }

    // カウンタ値を取得（first: タグ比較回数, second: 文字列比較回数）
    std::pair<std::uint64_t, std::uint64_t> counters() const {
        return {tag_comparisons_, string_comparisons_};
    }

    // FNV-1a 32bit ハッシュ値を計算
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

    // n が素数かどうか判定
    static bool is_prime(int n) {
        if (n < 2) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        for (int i = 3; i * i <= n; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    // 現在のバケット数の2倍以上で最小の素数を返す
    static int next_prime(int current) {
        int candidate = current * 2 + 1;
        while (!is_prime(candidate)) {
            candidate += 2;
        }
        return candidate;
    }

    // バケット数を拡張して全要素を再配置（保存済みハッシュ値を再利用）
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

    // 負荷率が 1.0 を超えたら自動拡張して先頭挿入
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

    // key に対応する値へのポインタを返す（見つからなければ nullptr）
    // タグ（保存ハッシュ値）が一致した場合のみ文字列比較を行う
    const T* operator()(const std::string& key) const {
        std::uint32_t hval = fnv1a(key);
        int h = static_cast<int>(hval % static_cast<std::uint32_t>(bucket_count_));
        Node* p = buckets_[h];
        while (p != nullptr) {
            ++tag_comparisons_;
            if (p->hash == hval) {
                ++string_comparisons_;
                if (p->key == key) {
                    return &(p->value);
                }
            }
            p = p->next;
        }
        return nullptr;
    }

    // key を削除する（存在しなければ何もしない）
    HashTable& operator-=(const std::string& key) {
        std::uint32_t hval = fnv1a(key);
        int h = static_cast<int>(hval % static_cast<std::uint32_t>(bucket_count_));
        Node* p = buckets_[h];
        Node* prev = nullptr;
        while (p != nullptr) {
            ++tag_comparisons_;
            if (p->hash == hval) {
                ++string_comparisons_;
                if (p->key == key) {
                    if (prev == nullptr) {
                        buckets_[h] = p->next;
                    } else {
                        prev->next = p->next;
                    }
                    delete p;
                    --size_;
                    return *this;
                }
            }
            prev = p;
            p = p->next;
        }
        return *this;
    }

    // 要素数を返す
    int size() const {
        return size_;
    }

    // バケット数を返す
    int bucket_count() const {
        return bucket_count_;
    }

    // 負荷率を返す
    double load_factor() const {
        return static_cast<double>(size_) / static_cast<double>(bucket_count_);
    }

    // 全バケットの内容を表示（デバッグ用、max_buckets >= 0 なら先頭のみ）
    void dump(int max_buckets = -1) const {
        int limit = (max_buckets >= 0 && max_buckets < bucket_count_) ? max_buckets : bucket_count_;
        for (int i = 0; i < limit; ++i) {
            std::cout << "  bucket[" << i << "]: ";
            Node* p = buckets_[i];
            if (p == nullptr) {
                std::cout << "(empty)";
            }
            while (p != nullptr) {
                std::cout << "(" << p->key << " -> " << p->value << ")";
                if (p->next != nullptr) {
                    std::cout << " -> ";
                }
                p = p->next;
            }
            std::cout << std::endl;
        }
        if (limit < bucket_count_) {
            std::cout << "  ... (残り " << (bucket_count_ - limit) << " バケットは省略)" << std::endl;
        }
    }

private:
    Node** buckets_;
    int bucket_count_;
    int size_;
    mutable std::uint64_t tag_comparisons_;
    mutable std::uint64_t string_comparisons_;
};

int main() {
    HashTable<int> ht;

    std::cout << "=== 大量挿入前 ===" << std::endl;
    std::cout << "bucket_count: " << ht.bucket_count() << ", size: " << ht.size()
              << ", load_factor: " << ht.load_factor() << std::endl;

    std::cout << "=== key0000 〜 key0999 を 0 〜 999 とともに挿入 ===" << std::endl;
    int prev_bucket_count = ht.bucket_count();
    for (int i = 0; i < 1000; ++i) {
        std::string key = "key";
        if (i < 10) {
            key += "000";
        } else if (i < 100) {
            key += "00";
        } else if (i < 1000) {
            key += "0";
        }
        key += std::to_string(i);
        ht.insert(key, i);
        if (ht.bucket_count() != prev_bucket_count) {
            std::cout << "  拡張: bucket_count " << prev_bucket_count << " -> " << ht.bucket_count()
                      << " (size=" << ht.size() << ", load_factor=" << ht.load_factor() << ")" << std::endl;
            prev_bucket_count = ht.bucket_count();
        }
    }

    std::cout << "=== 挿入後 ===" << std::endl;
    std::cout << "bucket_count: " << ht.bucket_count() << ", size: " << ht.size()
              << ", load_factor: " << ht.load_factor() << std::endl;

    std::cout << "=== 検索テスト ===" << std::endl;
    const int* p;
    p = ht("key0000");
    std::cout << "find key0000: " << (p != nullptr ? std::to_string(*p) : "none") << std::endl;
    p = ht("key0123");
    std::cout << "find key0123: " << (p != nullptr ? std::to_string(*p) : "none") << std::endl;
    p = ht("key0999");
    std::cout << "find key0999: " << (p != nullptr ? std::to_string(*p) : "none") << std::endl;
    p = ht("key1000");
    std::cout << "find key1000: " << (p != nullptr ? std::to_string(*p) : "none") << std::endl;

    std::cout << "=== 全キー検索でのタグ照合効果 ===" << std::endl;
    ht.reset_counters();
    for (int i = 0; i < 1000; ++i) {
        std::string key = "key";
        if (i < 10) {
            key += "000";
        } else if (i < 100) {
            key += "00";
        } else if (i < 1000) {
            key += "0";
        }
        key += std::to_string(i);
        const int* v = ht(key);
        if (v == nullptr) {
            std::cerr << "unexpected missing key: " << key << std::endl;
        }
    }
    auto c = ht.counters();
    std::cout << "1000 件全キー検索: tag comparisons = " << c.first
              << ", string comparisons = " << c.second << std::endl;
    std::cout << "  -> タグ不一致により文字列比較を " << (c.first - c.second)
              << " 回省略しました" << std::endl;

    std::cout << "=== 削除テスト ===" << std::endl;
    ht -= "key0123";
    std::cout << "key0123 を削除しました" << std::endl;
    p = ht("key0123");
    std::cout << "find key0123 after erase: " << (p != nullptr ? std::to_string(*p) : "none") << std::endl;
    std::cout << "size: " << ht.size() << ", load_factor: " << ht.load_factor() << std::endl;

    std::cout << "=== 先頭 30 バケットの内容（拡張後も整合性確認） ===" << std::endl;
    ht.dump(30);

    return 0;
}
