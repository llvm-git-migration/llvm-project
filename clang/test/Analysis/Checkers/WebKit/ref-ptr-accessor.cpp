// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

#include "ref.h"

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
