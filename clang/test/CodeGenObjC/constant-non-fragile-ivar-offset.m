// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -emit-llvm %s -o - | FileCheck %s

// CHECK: @"OBJC_IVAR_$_StaticLayout.static_layout_ivar" = hidden constant i64 20
// CHECK: @"OBJC_IVAR_$_SuperClass.superClassIvar" = hidden constant i64 20
// CHECK: @"OBJC_IVAR_$_SuperClass._superClassProperty" = hidden constant i64 24
// CHECK: @"OBJC_IVAR_$_IntermediateClass.intermediateClassIvar" = constant i64 32
// CHECK: @"OBJC_IVAR_$_IntermediateClass.intermediateClassIvar2" = constant i64 40
// CHECK: @"OBJC_IVAR_$_IntermediateClass._intermediateProperty" = hidden constant i64 48
// CHECK: @"OBJC_IVAR_$_SubClass.subClassIvar" = constant i64 56
// CHECK: @"OBJC_IVAR_$_NotStaticLayout.not_static_layout_ivar" = hidden global i64 12

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
  // CHECK: load i64, ptr @"OBJC_IVAR_$_StaticLayout
}
@end

// Ivars declared in the @interface
@interface SuperClass : NSObject
@property (nonatomic, assign) int superClassProperty;
@end

@implementation SuperClass {
  int superClassIvar; // Declare an ivar
}
- (void)superClassMethod {
    _superClassProperty = 42;
    superClassIvar = 10;
    // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_SuperClass
    // CHECK: getelementptr inbounds i8, ptr @"OBJC_IVAR_$_SuperClass
}
@end

// Inheritance and Ivars
@interface IntermediateClass : SuperClass
{
    double intermediateClassIvar;

    @protected
    int intermediateClassIvar2;
}
@property (nonatomic, strong) SuperClass *intermediateProperty;
@end

@implementation IntermediateClass
@synthesize intermediateProperty = _intermediateProperty;
- (void)intermediateClassMethod {
    intermediateClassIvar = 3.14;
    // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_IntermediateClass
    // CHECK: getelementptr inbounds i8, ptr @"OBJC_IVAR_$_IntermediateClass
}

- (void)intermediateClassPropertyMethod {
    self.intermediateProperty = 0;
    // CHECK: getelementptr inbounds i8, ptr @"OBJC_IVAR_$_IntermediateClass._intermediateProperty"
}
@end

@interface SubClass : IntermediateClass
{
    double subClassIvar;
}
@end

@implementation SubClass
- (void)subclassVar {
    
    subClassIvar = 6.28;
    // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_SubClass
    // CHECK: getelementptr inbounds i8, ptr @"OBJC_IVAR_$_SubClass
}

-(void)intermediateSubclassVar
{
    intermediateClassIvar = 3.14;
    // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_IntermediateClass
    // CHECK: getelementptr inbounds i8, ptr @"OBJC_IVAR_$_IntermediateClass
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
  // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_NotStaticLayout.not_static_layout_ivar
}
@end
