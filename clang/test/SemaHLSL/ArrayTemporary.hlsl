// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ast-dump %s | Filecheck %s

void fn(float x[2]) { }

// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float *)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (float *)' lvalue Function {{.*}} 'fn' 'void (float *)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float *' <ArrayToPointerDecay>
// CHECK-NEXT: HLSLArrayTemporaryExpr {{.*}} 'float[2]' lvalue

void call() {
  float Arr[2] = {0, 0};
  fn(Arr);
}

struct Obj {
  float V;
  int X;
};

void fn2(Obj O[4]) { }

// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(Obj *)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (Obj *)' lvalue Function {{.*}} 'fn2' 'void (Obj *)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'Obj *' <ArrayToPointerDecay>
// CHECK-NEXT: HLSLArrayTemporaryExpr {{.*}} 'Obj[4]' lvalue

void call2() {
  Obj Arr[4] = {};
  fn2(Arr);
}


void fn3(float x[2][2]) { }

// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float (*)[2])' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (float (*)[2])' lvalue Function {{.*}} 'fn3' 'void (float (*)[2])'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: HLSLArrayTemporaryExpr {{.*}} 'float[2][2]' lvalue

void call3() {
  float Arr[2][2] = {{0, 0}, {1,1}};
  fn3(Arr);
}
