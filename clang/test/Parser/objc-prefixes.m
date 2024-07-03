// RUN: %clang_cc1 %s -Wobjc-prefix-length=2 -fsyntax-only -verify

// Test prefix length rules for ObjC interfaces and protocols

// -- Plain interfaces --------------------------------------------------------

@interface _Foo
@end

@interface Foo // expected-warning {{un-prefixed Objective-C class name}}
@end

@interface NSFoo
@end

// Special case for prefix-length 2
@interface NSCFFoo
@end

@interface NSCFXFoo // expected-warning {{un-prefixed Objective-C class name}}
@end

@interface NSXFoo // expected-warning {{un-prefixed Objective-C class name}}
@end

// -- Categories --------------------------------------------------------------

// Categories don't trigger these warnings

@interface Foo (Bar)
@end

@interface NSFoo (Bar)
@end

@interface NSCFFoo (Bar)
@end

@interface NSXFoo (Bar)
@end

// -- Protocols ---------------------------------------------------------------

@protocol _FooProtocol
@end

@protocol FooProtocol // expected-warning {{un-prefixed Objective-C protocol name}}
@end

@protocol NSFooProtocol
@end

// Special case for prefix-length 2
@protocol NSCFFooProtocol
@end

@protocol NSCFXFooProtocol // expected-warning {{un-prefixed Objective-C protocol name}}
@end

@protocol NSXFooProtocol // expected-warning {{un-prefixed Objective-C protocol name}}
@end

