#include <iostream>
#include <string>

// hash.c をそのまま C++ class に移植した非テンプレート版（値は int 固定）
class HashTable {
public:
    static constexpr int HASHSIZE = 17;

    // チェインのノード
    struct Node {
        std::string key;
        int value;
        Node* next;
    };

    // 全バケットを nullptr で初期化
    HashTable() {
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
    bool insert(const std::string& key, int value) {
        int h = hash(key);
        Node* p = new Node{key, value, buckets_[h]};
        buckets_[h] = p;
        return true;
    }

    // key に対応する値へのポインタを返す（見つからなければ nullptr）
    const int* operator()(const std::string& key) const {
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
                return *this;
            }
            prev = p;
            p = p->next;
        }
        return *this;
    }

private:
    Node* buckets_[HASHSIZE];
};

int main() {
    HashTable ht;

    std::cout << "--- insert ---" << std::endl;
    ht.insert("takimoto", 42);
    ht.insert("katsurada", 122);
    ht.insert("matsuzawa", 35);
    ht.insert("ohmura", 12);
    std::cout << "takimoto=42, katsurada=122, matsuzawa=35, ohmura=12 を挿入しました" << std::endl;

    std::cout << "--- find (success) ---" << std::endl;
    const int* p;
    p = ht("takimoto");
    std::cout << "takimoto: " << (p != nullptr ? std::to_string(*p) : "none") << std::endl;
    p = ht("katsurada");
    std::cout << "katsurada: " << (p != nullptr ? std::to_string(*p) : "none") << std::endl;
    p = ht("matsuzawa");
    std::cout << "matsuzawa: " << (p != nullptr ? std::to_string(*p) : "none") << std::endl;
    p = ht("ohmura");
    std::cout << "ohmura: " << (p != nullptr ? std::to_string(*p) : "none") << std::endl;

    std::cout << "--- find (failure) ---" << std::endl;
    p = ht("nonexistent");
    std::cout << "nonexistent: " << (p != nullptr ? std::to_string(*p) : "none") << std::endl;

    std::cout << "--- erase ---" << std::endl;
    ht -= "katsurada";
    std::cout << "katsurada を削除しました" << std::endl;
    p = ht("katsurada");
    std::cout << "katsurada (after erase): " << (p != nullptr ? std::to_string(*p) : "none") << std::endl;

    p = ht("takimoto");
    std::cout << "takimoto (after erase): " << (p != nullptr ? std::to_string(*p) : "none") << std::endl;

    return 0;
}
