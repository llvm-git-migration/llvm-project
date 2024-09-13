; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=none \
; RUN: -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=NO-VP %s

define void @vp_icmp(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL->NEXT: Live-in vp<%0> = VF * UF
; IF-EVL->NEXT: Live-in vp<%1> = vector-trip-count
; IF-EVL->NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%12>
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%4> = phi ir<0>, vp<%11>
; IF-EVL-NEXT:     EMIT vp<%5> = EXPLICIT-VECTOR-LENGTH vp<%4>, ir<%N>
; IF-EVL-NEXT:     vp<%6> = SCALAR-STEPS vp<%4>, ir<1>
; IF-EVL-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%6>
; IF-EVL-NEXT:     vp<%7> = vector-pointer ir<%arrayidx>
; IF-EVL-NEXT:     WIDEN ir<%0> = vp.load vp<%7>, vp<%5>
; IF-EVL-NEXT:     CLONE ir<%arrayidx3> = getelementptr inbounds ir<%c>, vp<%6>
; IF-EVL-NEXT:     vp<%8> = vector-pointer ir<%arrayidx3>
; IF-EVL-NEXT:     WIDEN ir<%1> = vp.load vp<%8>, vp<%5>
; IF-EVL-NEXT:     WIDEN ir<%cmp4> = vp.icmp sgt ir<%0>, ir<%1>, vp<%5>
; IF-EVL-NEXT:     WIDEN-CAST ir<%conv5> = zext  ir<%cmp4> to i32
; IF-EVL-NEXT:     CLONE ir<%arrayidx7> = getelementptr inbounds ir<%a>, vp<%6>
; IF-EVL-NEXT:     vp<%9> = vector-pointer ir<%arrayidx7>
; IF-EVL-NEXT:     WIDEN vp.store vp<%9>, ir<%conv5>, vp<%5>
; IF-EVL-NEXT:     SCALAR-CAST vp<%10> = zext vp<%5> to i64
; IF-EVL-NEXT:     EMIT vp<%11> = add vp<%10>, vp<%4>
; IF-EVL-NEXT:     EMIT vp<%12> = add vp<%3>, vp<%0>
; IF-EVL-NEXT:     EMIT branch-on-count vp<%12>, vp<%1>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

; NO-VP: Plan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF>=1' {
; NO-VP->NEXT: Live-in vp<%0> = VF * UF
; NO-VP->NEXT: Live-in vp<%1> = vector-trip-count
; NO-VP->NEXT: Live-in ir<%N> = original trip-count

; NO-VP: vector.ph:
; NO-VP-NEXT: Successor(s): vector loop

; NO-VP: <x1> vector loop: {
; NO-VP-NEXT:   vector.body:
; NO-VP-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%7>
; NO-VP-NEXT:     vp<%3> = SCALAR-STEPS vp<%2>, ir<1>
; NO-VP-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%3>
; NO-VP-NEXT:     vp<%4> = vector-pointer ir<%arrayidx>
; NO-VP-NEXT:     WIDEN ir<%0> = load vp<%4>
; NO-VP-NEXT:     CLONE ir<%arrayidx3> = getelementptr inbounds ir<%c>, vp<%3>
; NO-VP-NEXT:     vp<%5> = vector-pointer ir<%arrayidx3>
; NO-VP-NEXT:     WIDEN ir<%1> = load vp<%5>
; NO-VP-NEXT:     WIDEN ir<%cmp4> = icmp sgt ir<%0>, ir<%1>
; NO-VP-NEXT:     WIDEN-CAST ir<%conv5> = zext  ir<%cmp4> to i32
; NO-VP-NEXT:     CLONE ir<%arrayidx7> = getelementptr inbounds ir<%a>, vp<%3>
; NO-VP-NEXT:     vp<%6> = vector-pointer ir<%arrayidx7>
; NO-VP-NEXT:     WIDEN store vp<%6>, ir<%conv5>
; NO-VP-NEXT:     EMIT vp<%7> = add nuw vp<%2>, vp<%0>
; NO-VP-NEXT:     EMIT branch-on-count vp<%7>, vp<%1>
; NO-VP-NEXT:   No successors
; NO-VP-NEXT: }

entry:
  %cmp12 = icmp sgt i64 %N, 0
  br i1 %cmp12, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx3, align 4
  %cmp4 = icmp sgt i32 %0, %1
  %conv5 = zext i1 %cmp4 to i32
  %arrayidx7 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %conv5, ptr %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @vp_fcmp(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL->NEXT: Live-in vp<%0> = VF * UF
; IF-EVL->NEXT: Live-in vp<%1> = vector-trip-count
; IF-EVL->NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%12>
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%4> = phi ir<0>, vp<%11>
; IF-EVL-NEXT:     EMIT vp<%5> = EXPLICIT-VECTOR-LENGTH vp<%4>, ir<%N>
; IF-EVL-NEXT:     vp<%6> = SCALAR-STEPS vp<%4>, ir<1>
; IF-EVL-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%6>
; IF-EVL-NEXT:     vp<%7> = vector-pointer ir<%arrayidx>
; IF-EVL-NEXT:     WIDEN ir<%0> = vp.load vp<%7>, vp<%5>
; IF-EVL-NEXT:     CLONE ir<%arrayidx3> = getelementptr inbounds ir<%c>, vp<%6>
; IF-EVL-NEXT:     vp<%8> = vector-pointer ir<%arrayidx3>
; IF-EVL-NEXT:     WIDEN ir<%1> = vp.load vp<%8>, vp<%5>
; IF-EVL-NEXT:     WIDEN ir<%cmp4> = vp.fcmp ogt ir<%0>, ir<%1>, vp<%5>
; IF-EVL-NEXT:     WIDEN-CAST ir<%conv6> = uitofp  ir<%cmp4> to float
; IF-EVL-NEXT:     CLONE ir<%arrayidx8> = getelementptr inbounds ir<%a>, vp<%6>
; IF-EVL-NEXT:     vp<%9> = vector-pointer ir<%arrayidx8>
; IF-EVL-NEXT:     WIDEN vp.store vp<%9>, ir<%conv6>, vp<%5>
; IF-EVL-NEXT:     SCALAR-CAST vp<%10> = zext vp<%5> to i64
; IF-EVL-NEXT:     EMIT vp<%11> = add vp<%10>, vp<%4>
; IF-EVL-NEXT:     EMIT vp<%12> = add vp<%3>, vp<%0>
; IF-EVL-NEXT:     EMIT branch-on-count vp<%12>, vp<%1>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

; NO-VP: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF>=1' {
; NO-VP->NEXT: Live-in vp<%0> = VF * UF
; NO-VP->NEXT: Live-in vp<%1> = vector-trip-count
; NO-VP->NEXT: Live-in ir<%N> = original trip-count

; NO-VP: vector.ph:
; NO-VP->NEXT: Successor(s): vector loop

; NO-VP: <x1> vector loop: {
; NO-VP->NEXT:   vector.body:
; NO-VP->NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%7>
; NO-VP->NEXT:     vp<%3> = SCALAR-STEPS vp<%2>, ir<1>
; NO-VP->NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%3>
; NO-VP->NEXT:     vp<%4> = vector-pointer ir<%arrayidx>
; NO-VP->NEXT:     WIDEN ir<%0> = load vp<%4>
; NO-VP->NEXT:     CLONE ir<%arrayidx3> = getelementptr inbounds ir<%c>, vp<%3>
; NO-VP->NEXT:     vp<%5> = vector-pointer ir<%arrayidx3>
; NO-VP->NEXT:     WIDEN ir<%1> = load vp<%5>
; NO-VP->NEXT:     WIDEN ir<%cmp4> = fcmp ogt ir<%0>, ir<%1>
; NO-VP->NEXT:     WIDEN-CAST ir<%conv6> = uitofp  ir<%cmp4> to float
; NO-VP->NEXT:     CLONE ir<%arrayidx8> = getelementptr inbounds ir<%a>, vp<%3>
; NO-VP->NEXT:     vp<%6> = vector-pointer ir<%arrayidx8>
; NO-VP->NEXT:     WIDEN store vp<%6>, ir<%conv6>
; NO-VP->NEXT:     EMIT vp<%7> = add nuw vp<%2>, vp<%0>
; NO-VP->NEXT:     EMIT branch-on-count vp<%7>, vp<%1>
; NO-VP->NEXT:   No successors
; NO-VP->NEXT: }

entry:
  %cmp13 = icmp sgt i64 %N, 0
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds float, ptr %c, i64 %indvars.iv
  %1 = load float, ptr %arrayidx3, align 4
  %cmp4 = fcmp ogt float %0, %1
  %conv6 = uitofp i1 %cmp4 to float
  %arrayidx8 = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  store float %conv6, ptr %arrayidx8, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}