#include <iostream>
#include <string>

// キーは std::string 固定、値は任意型 T のテンプレート版ハッシュ表（チェイン法）
template <class T>
class HashTable {
public:
    static constexpr int HASHSIZE = 17;

    // チェインのノード
    struct Node {
        std::string key;
        T value;
        Node* next;
    };

    // 全バケットを nullptr で初期化
    HashTable() : size_(0) {
        for (int i = 0; i < HASHSIZE; ++i) {
            buckets_[i] = nullptr;
        }
    }

    // 全ノードを削除
    ~HashTable() {
        for (int i = 0; i < HASHSIZE; ++i) {
            Node* p = buckets_[i];
            while (p != nullptr) {
                Node* next = p->next;
                delete p;
                p = next;
            }
        }
    }

    // コピー禁止
    HashTable(const HashTable&) = delete;
    HashTable& operator=(const HashTable&) = delete;

    // ハッシュ値を計算（文字コードの総和 % HASHSIZE）
    static int hash(const std::string& key) {
        int hashval = 0;
        for (char c : key) {
            hashval += static_cast<unsigned char>(c);
        }
        return hashval % HASHSIZE;
    }

    // 先頭挿入で key-value を登録
    bool insert(const std::string& key, const T& value) {
        int h = hash(key);
        Node* p = new Node{key, value, buckets_[h]};
        buckets_[h] = p;
        ++size_;
        return true;
    }

    // key に対応する値へのポインタを返す（見つからなければ nullptr）
    const T* operator()(const std::string& key) const {
        int h = hash(key);
        Node* p = buckets_[h];
        while (p != nullptr) {
            if (p->key == key) {
                return &(p->value);
            }
            p = p->next;
        }
        return nullptr;
    }

    // key を削除する（存在しなければ何もしない）
    HashTable& operator-=(const std::string& key) {
        int h = hash(key);
        Node* p = buckets_[h];
        Node* prev = nullptr;
        while (p != nullptr) {
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
            prev = p;
            p = p->next;
        }
        return *this;
    }

    // 登録されている要素数を返す
    int size() const {
        return size_;
    }

    // 全バケットの内容を表示（デバッグ用）
    void dump() const {
        for (int i = 0; i < HASHSIZE; ++i) {
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
    }

private:
    Node* buckets_[HASHSIZE];
    int size_;
};

int main() {
    std::cout << "=== (a) HashTable<int> ===" << std::endl;
    HashTable<int> ht_int;
    ht_int.insert("takimoto", 42);
    ht_int.insert("katsurada", 122);
    ht_int.insert("matsuzawa", 35);
    ht_int.insert("ohmura", 12);
    std::cout << "insert 後の size: " << ht_int.size() << std::endl;
    ht_int.dump();

    const int* pi = ht_int("takimoto");
    std::cout << "find takimoto: " << (pi != nullptr ? std::to_string(*pi) : "none") << std::endl;
    pi = ht_int("katsurada");
    std::cout << "find katsurada: " << (pi != nullptr ? std::to_string(*pi) : "none") << std::endl;
    pi = ht_int("nonexistent");
    std::cout << "find nonexistent: " << (pi != nullptr ? std::to_string(*pi) : "none") << std::endl;

    ht_int -= "katsurada";
    std::cout << "katsurada 削除後の size: " << ht_int.size() << std::endl;
    pi = ht_int("katsurada");
    std::cout << "find katsurada after erase: " << (pi != nullptr ? std::to_string(*pi) : "none") << std::endl;

    std::cout << std::endl << "=== (b) HashTable<double> ===" << std::endl;
    HashTable<double> ht_double;
    ht_double.insert("takimoto", 172.5);
    ht_double.insert("katsurada", 168.2);
    ht_double.insert("matsuzawa", 175.0);
    std::cout << "insert 後の size: " << ht_double.size() << std::endl;
    const double* pd = ht_double("matsuzawa");
    std::cout << "find matsuzawa: " << (pd != nullptr ? std::to_string(*pd) : "none") << std::endl;
    pd = ht_double("ohmura");
    std::cout << "find ohmura: " << (pd != nullptr ? std::to_string(*pd) : "none") << std::endl;

    std::cout << std::endl << "=== (c) HashTable<std::string> ===" << std::endl;
    HashTable<std::string> ht_str;
    ht_str.insert("greeting", "こんにちは");
    ht_str.insert("subject", "情報科学演習");
    ht_str.insert("name", "哈希表");
    std::cout << "insert 後の size: " << ht_str.size() << std::endl;
    const std::string* ps = ht_str("greeting");
    std::cout << "find greeting: " << (ps != nullptr ? *ps : "none") << std::endl;
    ps = ht_str("subject");
    std::cout << "find subject: " << (ps != nullptr ? *ps : "none") << std::endl;
    ps = ht_str("farewell");
    std::cout << "find farewell: " << (ps != nullptr ? *ps : "none") << std::endl;

    std::cout << std::endl << "=== (d) 存在しないキーと削除後の検索 ===" << std::endl;
    HashTable<int> ht_check;
    ht_check.insert("present", 99);
    const int* pc = ht_check("absent");
    std::cout << "find absent (before erase present): " << (pc != nullptr ? std::to_string(*pc) : "none") << std::endl;
    ht_check -= "present";
    pc = ht_check("present");
    std::cout << "find present (after erase present): " << (pc != nullptr ? std::to_string(*pc) : "none") << std::endl;
    pc = ht_check("absent");
    std::cout << "find absent (after erase present): " << (pc != nullptr ? std::to_string(*pc) : "none") << std::endl;

    return 0;
}
