// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

template<typename T>
class Ref {
public:
    ~Ref()
    {
        if (auto* ptr = m_ptr)
            ptr->deref();
        m_ptr = nullptr;
    }

    Ref(T& object)
        : m_ptr(&object)
    {
        object.ref();
    }

    Ref(const Ref& other)
        : m_ptr(other.ptr())
    {
        m_ptr->ref();
    }

    template<typename X> Ref(const Ref<X>& other)
        : m_ptr(other.ptr())
    {
        m_ptr->ref();
    }

    Ref(Ref&& other)
        : m_ptr(&other.leakRef())
    {
    }

    template<typename X>
    Ref(Ref<X>&& other)
        : m_ptr(&other.leakRef())
    {
    }

    T* operator->() const { return m_ptr; }
    T* ptr() const { return m_ptr; }
    T& get() const { return *m_ptr; }
    operator T&() const { return *m_ptr; }
    bool operator!() const { return !*m_ptr; }

    T& leakRef()
    {
        T& result = *m_ptr;
        m_ptr = nullptr;
        return result;
    }

private:
    template<typename U> friend inline Ref<U> adoptRef(U&);

    enum AdoptTag { Adopt };
    Ref(T& object, AdoptTag)
        : m_ptr(&object)
    {
    }

    T* m_ptr;
};

class Node {
public:
    static Ref<Node> create();
    virtual ~Node();

    void ref() const;
    void deref() const;

protected:
    explicit Node();
};

void someFunction(Node*);

void testFunction()
{
    Ref node = Node::create();
    someFunction(node.ptr());
}
