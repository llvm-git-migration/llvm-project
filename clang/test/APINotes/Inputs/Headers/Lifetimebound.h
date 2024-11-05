int *funcToAnnotate(int *p);

// TODO: support annotating ctors.
struct MyClass {
    MyClass(int*);
    int *annotateThis();
    int *methodToAnnotate(int *p);
};
