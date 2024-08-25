// RUN: %clang_dxc -T cs_6_0 %s | Filecheck %s

// Verify that internal linkage unused functions are removed

RWBuffer<unsigned> buf;

// CHECK-NOT: define{{.*}}donothing
void donothing() {
     buf[1] = 1; // never called, does nothing!
}


[numthreads(1,1,1)]
[shader("compute")]
void main() {
     buf[0] = 0;// I'm doing something!!!     
}