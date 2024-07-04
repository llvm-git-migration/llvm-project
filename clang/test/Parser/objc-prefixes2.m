// RUN: %clang_cc1 %s -Wobjc-prefixes=NS,NSCF,NSURL -Wobjc-forbidden-prefixes=XX -fsyntax-only -verify

// Test prefix list rules

@interface Foo // expected-warning {{Objective-C class name prefix not in permitted list}}
@end

@interface NSFoo
@end

@interface NSfoo // expected-warning {{Objective-C class name prefix not in permitted list}}
@end

@interface NSFFoo // expected-warning {{Objective-C class name prefix not in permitted list}}
@end

@interface NSCFFoo
@end

@interface NSURL
@end

@interface NSURLFoo
@end

@interface NSRGBColor // expected-warning {{Objective-C class name prefix not in permitted list}}
@end

@protocol NSRGBColorProtocol // expected-warning {{Objective-C protocol name prefix not in permitted list}}
@end

@interface XXFoo // expected-warning {{Objective-C class name prefix in forbidden list}}
@end

@protocol XXFooProtocol // expected-warning {{Objective-C protocol name prefix in forbidden list}}
@end
