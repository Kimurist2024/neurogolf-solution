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

static int total_tests = 0;
static int passed_tests = 0;

static void CHECK(bool cond, const char* name) {
    ++total_tests;
    if (cond) {
        ++passed_tests;
        std::cout << "OK: " << name << "\n";
    } else {
        std::cout << "NG: " << name << "\n";
    }
}

int main() {
    // 1. empty table
    {
        HashTable<int> ht;
        CHECK(ht("missing") == nullptr, "empty lookup returns nullptr");
        CHECK(ht.size() == 0, "empty size is 0");
        ht -= "missing";
        CHECK(ht.size() == 0, "erase on empty table is safe");
    }

    // 2. single insert and lookup
    {
        HashTable<int> ht;
        ht.insert("one", 1);
        const int* p = ht("one");
        CHECK(p != nullptr && *p == 1, "single insert/lookup");
    }

    // 3. colliding anagram keys (same additive hash)
    {
        HashTable<int> ht;
        ht.insert("abc", 1);
        ht.insert("acb", 2);
        ht.insert("bac", 3);
        ht.insert("bca", 4);
        ht.insert("cab", 5);
        ht.insert("cba", 6);
        const int* a = ht("abc");
        const int* b = ht("acb");
        const int* c = ht("bac");
        const int* d = ht("bca");
        const int* e = ht("cab");
        const int* f = ht("cba");
        CHECK(a && b && c && d && e && f, "all anagram keys found");
        CHECK(*a == 1 && *b == 2 && *c == 3 && *d == 4 && *e == 5 && *f == 6,
              "anagram values are correct");
    }

    // 4. erase head/middle/tail of a chain
    {
        HashTable<int> ht;
        ht.insert("abc", 1);
        ht.insert("acb", 2);
        ht.insert("bac", 3);
        // chain order after head-insertions: bac -> acb -> abc
        ht -= "bac";  // erase head
        CHECK(ht("bac") == nullptr, "erase head removes it");
        CHECK(ht("acb") != nullptr && *ht("acb") == 2, "head erase leaves middle");
        CHECK(ht("abc") != nullptr && *ht("abc") == 1, "head erase leaves tail");

        ht.insert("bac", 3);  // restore head
        ht -= "acb";          // erase middle
        CHECK(ht("acb") == nullptr, "erase middle removes it");
        CHECK(ht("bac") != nullptr && *ht("bac") == 3, "middle erase leaves head");
        CHECK(ht("abc") != nullptr && *ht("abc") == 1, "middle erase leaves tail");

        ht.insert("acb", 2);  // restore middle
        ht -= "abc";          // erase tail
        CHECK(ht("abc") == nullptr, "erase tail removes it");
        CHECK(ht("bac") != nullptr && *ht("bac") == 3, "tail erase leaves head");
        CHECK(ht("acb") != nullptr && *ht("acb") == 2, "tail erase leaves middle");
    }

    // 5. erasing non-existent key does not break the table
    {
        HashTable<int> ht;
        ht.insert("abc", 1);
        ht.insert("acb", 2);
        ht -= "notpresent";
        CHECK(ht.size() == 2, "erase missing key keeps size");
        CHECK(ht("abc") != nullptr && *ht("abc") == 1, "table intact after missing erase");
        CHECK(ht("acb") != nullptr && *ht("acb") == 2, "table intact after missing erase 2");
    }

    // 6. erase then re-insert
    {
        HashTable<int> ht;
        ht.insert("abc", 1);
        ht -= "abc";
        CHECK(ht("abc") == nullptr, "after erase key absent");
        ht.insert("abc", 10);
        const int* p = ht("abc");
        CHECK(p != nullptr && *p == 10, "re-inserted key found");
    }

    // 7. duplicate insert: front insert returns newest, one erase reveals old value
    {
        HashTable<int> ht;
        ht.insert("dup", 100);
        ht.insert("dup", 200);
        const int* p = ht("dup");
        CHECK(p != nullptr && *p == 200, "duplicate insert returns newer value");
        ht -= "dup";
        p = ht("dup");
        CHECK(p != nullptr && *p == 100, "one erase reveals old value");
    }

    // 8. size tracks inserts and erases
    {
        HashTable<int> ht;
        CHECK(ht.size() == 0, "initial size 0");
        ht.insert("a", 1);
        ht.insert("b", 2);
        CHECK(ht.size() == 2, "size after 2 inserts");
        ht -= "a";
        CHECK(ht.size() == 1, "size after 1 erase");
        ht -= "b";
        CHECK(ht.size() == 0, "size after all erased");
        ht -= "a";
        CHECK(ht.size() == 0, "size unchanged by missing erase");
    }

    // 9. bulk 1000 inserts/lookups and 500 erases
    {
        HashTable<int> ht;
        for (int i = 0; i < 1000; ++i) {
            ht.insert("key" + std::to_string(i), i);
        }
        bool all_found = true;
        for (int i = 0; i < 1000; ++i) {
            const int* p = ht("key" + std::to_string(i));
            if (p == nullptr || *p != i) {
                all_found = false;
                break;
            }
        }
        CHECK(all_found, "1000 inserted keys all found");

        for (int i = 0; i < 500; ++i) {
            ht -= "key" + std::to_string(i);
        }
        bool remaining_ok = true;
        for (int i = 500; i < 1000; ++i) {
            const int* p = ht("key" + std::to_string(i));
            if (p == nullptr || *p != i) {
                remaining_ok = false;
                break;
            }
        }
        bool erased_ok = true;
        for (int i = 0; i < 500; ++i) {
            if (ht("key" + std::to_string(i)) != nullptr) {
                erased_ok = false;
                break;
            }
        }
        CHECK(remaining_ok, "remaining 500 keys still found");
        CHECK(erased_ok, "erased 500 keys return nullptr");
        CHECK(ht.size() == 500, "size after 500 erases is 500");
    }

    // 10. HashTable<std::string> passes equivalents of tests 1-6
    {
        // 1 empty
        HashTable<std::string> ht;
        CHECK(ht("missing") == nullptr, "string empty lookup returns nullptr");
        CHECK(ht.size() == 0, "string empty size is 0");
        ht -= "missing";
        CHECK(ht.size() == 0, "string erase on empty table is safe");

        // 2 single insert
        ht.insert("greeting", "hello");
        const std::string* ps = ht("greeting");
        CHECK(ps != nullptr && *ps == "hello", "string single insert/lookup");

        // 3 anagram collisions
        ht.insert("abc", "A");
        ht.insert("acb", "B");
        ht.insert("bac", "C");
        ps = ht("abc");
        const std::string* ps2 = ht("acb");
        const std::string* ps3 = ht("bac");
        CHECK(ps && ps2 && ps3, "string anagram keys found");
        CHECK(*ps == "A" && *ps2 == "B" && *ps3 == "C", "string anagram values correct");

        // 4 erase head/middle/tail
        ht.insert("bca", "D");
        ht.insert("cab", "E");
        ht.insert("cba", "F");
        // current chain for this bucket: cba -> cab -> bca -> bac -> acb -> abc
        ht -= "cba";  // head of newly extended chain
        CHECK(ht("cba") == nullptr, "string erase head removes it");
        CHECK(ht("cab") != nullptr && *ht("cab") == "E", "string head erase leaves middle");
        CHECK(ht("bca") != nullptr && *ht("bca") == "D", "string head erase leaves tail-ish");

        ht -= "bca";  // tail of remaining chain
        CHECK(ht("bca") == nullptr, "string erase tail removes it");
        CHECK(ht("cab") != nullptr && *ht("cab") == "E", "string tail erase leaves head");
        CHECK(ht("bac") != nullptr && *ht("bac") == "C", "string tail erase leaves middle");

        ht -= "cab";  // middle of remaining chain
        CHECK(ht("cab") == nullptr, "string erase middle removes it");
        CHECK(ht("bac") != nullptr && *ht("bac") == "C", "string middle erase leaves one side");
        CHECK(ht("abc") != nullptr && *ht("abc") == "A", "string middle erase leaves other side");

        // 5 erase non-existent does not break table
        ht -= "notpresent";
        CHECK(ht("abc") != nullptr && *ht("abc") == "A", "string table intact after missing erase");
        CHECK(ht("bac") != nullptr && *ht("bac") == "C", "string table intact after missing erase 2");

        // 6 erase then re-insert
        ht -= "abc";
        CHECK(ht("abc") == nullptr, "string after erase key absent");
        ht.insert("abc", "newA");
        ps = ht("abc");
        CHECK(ps != nullptr && *ps == "newA", "string re-inserted key found");
    }

    std::cout << "passed " << passed_tests << " / " << total_tests << "\n";
    return (passed_tests == total_tests) ? 0 : 1;
}
