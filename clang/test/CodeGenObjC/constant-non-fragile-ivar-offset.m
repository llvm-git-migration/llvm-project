// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -emit-llvm %s -o - | FileCheck %s

// CHECK: @"OBJC_IVAR_$_StaticLayout.static_layout_ivar" = hidden constant i64 20
// CHECK: @"OBJC_IVAR_$_StaticLayoutSubClass.static_layout_ivar2" = hidden constant i64 24
// CHECK: @"OBJC_IVAR_$_MyClass.myIvar" = constant i64 20
// CHECK: @"OBJC_IVAR_$_MyClass._myProperty" = hidden constant i64 24
// CHECK: @"OBJC_IVAR_$_AnotherClass.privateId" = constant i64 24
// CHECK: @"OBJC_IVAR_$_AnotherClass.anotherPrivateId" = hidden constant i64 32
// CHECK: @"OBJC_IVAR_$_SuperClass.superClassIvar" = constant i64 20
// CHECK: @"OBJC_IVAR_$_SubClass.subClassIvar" = constant i64 24
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
  // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_StaticLayout
}
@end

// Scenario 1: Ivars declared in the @interface
@interface MyClass : NSObject
{
    int myIvar; // Declare an ivar
}

@property (nonatomic, assign) int myProperty; // Synthesize a property

@end

@implementation MyClass 

- (void)exampleMethod {
    self.myProperty = 42; // Access the property
    myIvar = 10; // Access the ivar directly
}

@end

// Scenario 2: Ivars declared directly in the @implementation
@interface AnotherClass : NSObject
{
  id privateId;
}

@end

@implementation AnotherClass
{
    id anotherPrivateId; // Declare an ivar directly in the implementation
}

- (void)doSomething {
   privateId = anotherPrivateId;
}

@end

// Scenario 3: Inheritance and Ivars
@interface SuperClass : NSObject
{
    int superClassIvar;
}
@end

@implementation SuperClass
@end

@interface SubClass : SuperClass
{
    double subClassIvar;
}
@end

@implementation SubClass
- (void)exampleMethod {
    // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$SuperClass
    superClassIvar = 100; // Access superclass ivar
    subClassIvar = 3.14; // Access subclass ivar
}
@end

// Scenario 4: Custom Getter/Setter Methods
@interface CustomPropertyClass : NSObject
@property (nonatomic, strong, getter=myCustomGetter, setter=myCustomSetter:) id customProperty;
@end

@implementation CustomPropertyClass
- (id) myCustomGetter {
    return 0;
}

- (void)myCustomSetter:(id)newValue {
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
