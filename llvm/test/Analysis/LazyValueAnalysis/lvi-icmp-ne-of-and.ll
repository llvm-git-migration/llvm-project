; RUN: opt < %s -passes="jump-threading,print<lazy-value-info>" -disable-output 2>&1 | FileCheck %s

; (Val & 4) != 4, sat
define i1 @test(i4 noundef %arg) {
entry:
  %mask = and i4 %arg, 4
  %cmp = icmp ne i4 %mask, 4
  br i1 %cmp, label %l_true, label %l_false

; CHECK-LABEL: l_true:
; CHECK-NEXT:  ; LatticeVal for: 'i4 %arg' is: constantrange<-8, 4>
l_true:
  ret i1 1
l_false:
  ret i1 0
}

; (Val & 5) != 4, sat
define i1 @test_2(i4 noundef %arg) {
entry:
  %mask = and i4 %arg, 5
  %cmp = icmp ne i4 %mask, 4
  br i1 %cmp, label %l_true, label %l_false

; CHECK-LABEL: l_true:
; CHECK-NEXT:  ; LatticeVal for: 'i4 %arg' is: constantrange<5, 4>
l_true:
  ret i1 1
l_false:
  ret i1 0
}

; (Val & 6) != 4, sat
define i1 @test_3(i4 noundef %arg) {
entry:
  %mask = and i4 %arg, 6
  %cmp = icmp ne i4 %mask, 4
  br i1 %cmp, label %l_true, label %l_false

; CHECK-LABEL: l_true:
; CHECK-NEXT:  ; LatticeVal for: 'i4 %arg' is: constantrange<6, 4>
l_true:
  ret i1 1
l_false:
  ret i1 0
}

; (Val & 8) != 4, unsat
define i1 @test_4(i4 noundef %arg) {
entry:
  %mask = and i4 %arg, 8
  %cmp = icmp ne i4 %mask, 4
  br i1 %cmp, label %l_true, label %l_false

; CHECK-LABEL: l_true:
; CHECK-NEXT:  ; LatticeVal for: 'i4 %arg' is: overdefined
l_true:
  ret i1 1
l_false:
  ret i1 0
}

; (Val & 11) != 6, unsat
define i1 @test_5(i4 noundef %arg) {
entry:
  %mask = and i4 %arg, 11
  %cmp = icmp ne i4 %mask, 6
  br i1 %cmp, label %l_true, label %l_false

; CHECK-LABEL: l_true:
; CHECK-NEXT:  ; LatticeVal for: 'i4 %arg' is: overdefined
l_true:
  ret i1 1
l_false:
  ret i1 0
}

; (Val & 8) != 16, unsat
define i1 @test_6(i8 noundef %arg) {
entry:
  %mask = and i8 %arg, 8
  %cmp = icmp ne i8 %mask, 16
  br i1 %cmp, label %l_true, label %l_false

; CHECK-LABEL: l_true:
; CHECK-NEXT:  ; LatticeVal for: 'i8 %arg' is: overdefined
l_true:
  ret i1 1
l_false:
  ret i1 0
}
