// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -emit-llvm %s -o - | FileCheck %s

// CHECK: @"OBJC_IVAR_$_StaticLayout.static_layout_ivar" = hidden constant i64 20
// CHECK: @"OBJC_IVAR_$_NotStaticLayout.not_static_layout_ivar" = hidden global i64 12
// CHECK: @"OBJC_IVAR_$_StaticLayoutSubClass.static_layout_ivar2" = hidden global i64 24

@interface NSObject {
  int these, will, never, change, ever;
}
@end

@interface StaticLayout : NSObject
@end

@implementation StaticLayout {
  int static_layout_ivar;
}
-(void)meth {
  static_layout_ivar = 0;
  // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_StaticLayout
}
@end

@interface StaticLayoutSubClass : StaticLayout
@end

@implementation StaticLayoutSubClass {
  int static_layout_ivar2;
}
-(void)meth2 {
  static_layout_ivar2 = 0;
  // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_StaticLayoutSubClass.static_layout_ivar2
}
@end

@interface NotNSObject {
  int these, might, change;
}
@end

@interface NotStaticLayout : NotNSObject
@end

@implementation NotStaticLayout {
  int not_static_layout_ivar;
}
-(void)meth {
  not_static_layout_ivar = 0;
  // CHECK: load i64, ptr @"OBJC_IVAR_$_NotStaticLayout.not_static_layout_ivar
}
@end
