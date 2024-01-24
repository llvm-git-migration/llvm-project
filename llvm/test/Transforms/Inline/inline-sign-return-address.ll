; Check the inliner doesn't inline a function with different sign return address schemes.
; RUN: opt < %s -passes=inline -S | FileCheck %s

declare void @init(ptr)

define internal i32 @foo_all() #0 {
  ret i32 43
}

define internal i32 @foo_nonleaf() #1 {
  ret i32 44
}

define internal i32 @foo_none() #2 {
  ret i32 42
}

define internal i32 @foo_lr() #3 {
  ret i32 45
}

define internal i32 @foo_bkey() #4 {
  ret i32 46
}

define dso_local i32 @bar_all() #0 {
; CHECK-LABEL: bar_all
; CHECK-NOT:   call i32 @foo_all()
; CHECK:       call i32 @foo_nonleaf()
; CHECK:       call i32 @foo_none()
; CHECK:       call i32 @foo_lr()
; CHECK:       call i32 @foo_bkey()
  %1 = call i32 @foo_all()
  %2 = call i32 @foo_nonleaf()
  %3 = call i32 @foo_none()
  %4 = call i32 @foo_lr()
  %5 = call i32 @foo_bkey()
  %6 = add nsw i32 %1, %2
  %7 = add nsw i32 %6, %3
  %8 = add nsw i32 %7, %4
  %9 = add nsw i32 %8, %5
  ret i32 %9
}

define dso_local i32 @bar_nonleaf() #1 {
; CHECK-LABEL: bar_nonleaf
; CHECK:       call i32 @foo_all()
; CHECK-NOT:   call i32 @foo_nonleaf()
; CHECK:       call i32 @foo_none()
; CHECK:       call i32 @foo_lr()
; CHECK:       call i32 @foo_bkey()
  %1 = call i32 @foo_all()
  %2 = call i32 @foo_nonleaf()
  %3 = call i32 @foo_none()
  %4 = call i32 @foo_lr()
  %5 = call i32 @foo_bkey()
  %6 = add nsw i32 %1, %2
  %7 = add nsw i32 %6, %3
  %8 = add nsw i32 %7, %4
  %9 = add nsw i32 %8, %5
  ret i32 %9
}

define dso_local i32 @bar_none() #2 {
; CHECK-LABEL: bar_none
; CHECK:       call i32 @foo_all()
; CHECK:       call i32 @foo_nonleaf()
; CHECK-NOT:   call i32 @foo_none()
; CHECK:       call i32 @foo_lr()
; CHECK:       call i32 @foo_bkey()
  %1 = call i32 @foo_all()
  %2 = call i32 @foo_nonleaf()
  %3 = call i32 @foo_none()
  %4 = call i32 @foo_lr()
  %5 = call i32 @foo_bkey()
  %6 = add nsw i32 %1, %2
  %7 = add nsw i32 %6, %3
  %8 = add nsw i32 %7, %4
  %9 = add nsw i32 %8, %5
  ret i32 %9
}

define dso_local i32 @bar_lr() #3 {
; CHECK-LABEL: bar_lr
; CHECK:       call i32 @foo_all()
; CHECK:       call i32 @foo_nonleaf()
; CHECK:       call i32 @foo_none()
; CHECK-NOT:   call i32 @foo_lr()
; CHECK:       call i32 @foo_bkey()
  %1 = call i32 @foo_all()
  %2 = call i32 @foo_nonleaf()
  %3 = call i32 @foo_none()
  %4 = call i32 @foo_lr()
  %5 = call i32 @foo_bkey()
  %6 = add nsw i32 %1, %2
  %7 = add nsw i32 %6, %3
  %8 = add nsw i32 %7, %4
  %9 = add nsw i32 %8, %5
  ret i32 %9
}

define dso_local i32 @bar_bkey() #4 {
; CHECK-LABEL: bar_bkey
; CHECK:       call i32 @foo_all()
; CHECK:       call i32 @foo_nonleaf()
; CHECK:       call i32 @foo_none()
; CHECK:       call i32 @foo_lr()
; CHECK-NOT:   call i32 @foo_bkey()
  %1 = call i32 @foo_all()
  %2 = call i32 @foo_nonleaf()
  %3 = call i32 @foo_none()
  %4 = call i32 @foo_lr()
  %5 = call i32 @foo_bkey()
  %6 = add nsw i32 %1, %2
  %7 = add nsw i32 %6, %3
  %8 = add nsw i32 %7, %4
  %9 = add nsw i32 %8, %5
  ret i32 %9
}


attributes #0 = { "branch-protection-pauth-lr"="false" "sign-return-address"="all" }
attributes #1 = { "branch-protection-pauth-lr"="false" "sign-return-address"="non-leaf" }
attributes #2 = { "branch-protection-pauth-lr"="false" "sign-return-address"="none" }
attributes #3 = { "branch-protection-pauth-lr"="true" "sign-return-address"="non-leaf" }
attributes #4 = { "branch-protection-pauth-lr"="true" "sign-return-address"="non-leaf" "sign-return-address-key"="b_key" }