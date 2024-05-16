; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S < %s | FileCheck %s --check-prefix=CHECK-VF4IC1 --check-prefix=CHECK
; RUN: opt -passes=loop-vectorize -force-vector-interleave=4 -force-vector-width=4 -S < %s | FileCheck %s --check-prefix=CHECK-VF4IC4 --check-prefix=CHECK
; RUN: opt -passes=loop-vectorize -force-vector-interleave=4 -force-vector-width=1 -S < %s | FileCheck %s --check-prefix=CHECK-VF1IC4 --check-prefix=CHECK

define i32 @select_const_i32_from_icmp(ptr nocapture readonly %v, i64 %n) {
; CHECK-LABEL: @select_const_i32_from_icmp
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <4 x i32>
; CHECK-VF4IC1-NEXT:   [[VEC_ICMP:%.*]] = icmp eq <4 x i32> [[VEC_LOAD]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC1-NEXT:   [[NOT:%.*]] = xor <4 x i1> [[VEC_ICMP]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 7, i32 3

; CHECK-VF4IC4:      vector.body:
; CHECK-VF4IC4:        [[VEC_PHI1:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL1:%.*]], %vector.body ]
; CHECK-VF4IC4-NEXT:   [[VEC_PHI2:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL2:%.*]], %vector.body ]
; CHECK-VF4IC4-NEXT:   [[VEC_PHI3:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL3:%.*]], %vector.body ]
; CHECK-VF4IC4-NEXT:   [[VEC_PHI4:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL4:%.*]], %vector.body ]
; CHECK-VF4IC4:        [[VEC_ICMP1:%.*]] = icmp eq <4 x i32> {{.*}}, <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC4-NEXT:   [[VEC_ICMP2:%.*]] = icmp eq <4 x i32> {{.*}}, <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC4-NEXT:   [[VEC_ICMP3:%.*]] = icmp eq <4 x i32> {{.*}}, <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC4-NEXT:   [[VEC_ICMP4:%.*]] = icmp eq <4 x i32> {{.*}}, <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC4-NEXT:   [[NOT1:%.*]] = xor <4 x i1> [[VEC_ICMP1]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC4-NEXT:   [[NOT2:%.*]] = xor <4 x i1> [[VEC_ICMP2]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC4-NEXT:   [[NOT3:%.*]] = xor <4 x i1> [[VEC_ICMP3]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC4-NEXT:   [[NOT4:%.*]] = xor <4 x i1> [[VEC_ICMP4]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC4-NEXT:   [[VEC_SEL1:%.*]] = or <4 x i1> [[VEC_PHI1]], [[NOT1]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL2:%.*]] = or <4 x i1> [[VEC_PHI2]], [[NOT2]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL3:%.*]] = or <4 x i1> [[VEC_PHI3]], [[NOT3]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL4:%.*]] = or <4 x i1> [[VEC_PHI4]], [[NOT4]]
; CHECK-VF4IC4:      middle.block:
; CHECK-VF4IC4-NEXT:   [[VEC_SEL5:%.*]] = or <4 x i1>  [[VEC_SEL2]], [[VEC_SEL1]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL6:%.*]] = or <4 x i1> [[VEC_SEL3]], [[VEC_SEL5]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL7:%.*]] = or <4 x i1> [[VEC_SEL4]], [[VEC_SEL6]]
; CHECK-VF4IC4-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL7]])
; CHECK-VF4IC4-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC4-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 7, i32 3


; CHECK-VF1IC4:      vector.body:
; CHECK-VF1IC4:        [[VEC_PHI1:%.*]] = phi i1 [ false, %vector.ph ], [ [[VEC_SEL1:%.*]], %vector.body ]
; CHECK-VF1IC4-NEXT:   [[VEC_PHI2:%.*]] = phi i1 [ false, %vector.ph ], [ [[VEC_SEL2:%.*]], %vector.body ]
; CHECK-VF1IC4-NEXT:   [[VEC_PHI3:%.*]] = phi i1 [ false, %vector.ph ], [ [[VEC_SEL3:%.*]], %vector.body ]
; CHECK-VF1IC4-NEXT:   [[VEC_PHI4:%.*]] = phi i1 [ false, %vector.ph ], [ [[VEC_SEL4:%.*]], %vector.body ]
; CHECK-VF1IC4:        [[VEC_LOAD1:%.*]] = load i32
; CHECK-VF1IC4-NEXT:   [[VEC_LOAD2:%.*]] = load i32
; CHECK-VF1IC4-NEXT:   [[VEC_LOAD3:%.*]] = load i32
; CHECK-VF1IC4-NEXT:   [[VEC_LOAD4:%.*]] = load i32
; CHECK-VF1IC4-NEXT:   [[VEC_ICMP1:%.*]] = icmp eq i32 [[VEC_LOAD1]], 3
; CHECK-VF1IC4-NEXT:   [[VEC_ICMP2:%.*]] = icmp eq i32 [[VEC_LOAD2]], 3
; CHECK-VF1IC4-NEXT:   [[VEC_ICMP3:%.*]] = icmp eq i32 [[VEC_LOAD3]], 3
; CHECK-VF1IC4-NEXT:   [[VEC_ICMP4:%.*]] = icmp eq i32 [[VEC_LOAD4]], 3
; CHECK-VF1IC4-NEXT:   [[NOT1:%.*]] = xor i1 [[VEC_ICMP1]], true
; CHECK-VF1IC4-NEXT:   [[NOT2:%.*]] = xor i1 [[VEC_ICMP2]], true
; CHECK-VF1IC4-NEXT:   [[NOT3:%.*]] = xor i1 [[VEC_ICMP3]], true
; CHECK-VF1IC4-NEXT:   [[NOT4:%.*]] = xor i1 [[VEC_ICMP4]], true
; CHECK-VF1IC4-NEXT:   [[VEC_SEL1:%.*]] = or i1 [[VEC_PHI1]], [[NOT1]]
; CHECK-VF1IC4-NEXT:   [[VEC_SEL2:%.*]] = or i1 [[VEC_PHI2]], [[NOT2]]
; CHECK-VF1IC4-NEXT:   [[VEC_SEL3:%.*]] = or i1 [[VEC_PHI3]], [[NOT3]]
; CHECK-VF1IC4-NEXT:   [[VEC_SEL4:%.*]] = or i1 [[VEC_PHI4]], [[NOT4]]
; CHECK-VF1IC4:      middle.block:
; CHECK-VF1IC4-NEXT:   [[VEC_SEL5:%.*]] = or i1 [[VEC_SEL2]], [[VEC_SEL1]]
; CHECK-VF1IC4-NEXT:   [[VEC_SEL6:%.*]] = or i1 [[VEC_SEL3]], [[VEC_SEL5]]
; CHECK-VF1IC4-NEXT:   [[OR_RDX:%.*]] = or i1  [[VEC_SEL4]], [[VEC_SEL6]]
; CHECK-VF1IC4-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF1IC4-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 7, i32 3

entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ 3, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select i1 %4, i32 %1, i32 7
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}


define i32 @select_const_i32_from_icmp2(ptr nocapture readonly %v, i64 %n) {
; CHECK-LABEL: @select_const_i32_from_icmp2
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <4 x i32>
; CHECK-VF4IC1-NEXT:   [[VEC_ICMP:%.*]] = icmp eq <4 x i32> [[VEC_LOAD]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <4 x i1> [[VEC_PHI]], [[VEC_ICMP]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 7, i32 3

entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ 3, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select i1 %4, i32 7, i32 %1
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}


define i32 @select_i32_from_icmp(ptr nocapture readonly %v, i32 %a, i32 %b, i64 %n) {
; CHECK-LABEL: @select_i32_from_icmp
; CHECK-VF4IC1:      vector.ph:
; CHECK-VF4IC1-NOT:    shufflevector <4 x i32>
; CHECK-VF4IC1-NOT:    shufflevector <4 x i32>
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <4 x i32>
; CHECK-VF4IC1-NEXT:   [[VEC_ICMP:%.*]] = icmp eq <4 x i32> [[VEC_LOAD]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC1-NEXT:   [[NOT:%.*]] = xor <4 x i1> [[VEC_ICMP]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 %b, i32 %a
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ %a, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select i1 %4, i32 %1, i32 %b
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}


define i32 @select_const_i32_from_fcmp_fast(ptr nocapture readonly %v, i64 %n) {
; CHECK-LABEL: @select_const_i32_from_fcmp_fast
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <4 x float>
; CHECK-VF4IC1-NEXT:   [[VEC_FCMP:%.*]] = fcmp fast ueq <4 x float> [[VEC_LOAD]], <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
; CHECK-VF4IC1-NEXT:   [[NOT:%.*]] = xor <4 x i1> [[VEC_FCMP]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 1, i32 2
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ 2, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds float, ptr %v, i64 %0
  %3 = load float, ptr %2, align 4
  %4 = fcmp fast ueq float %3, 3.0
  %5 = select i1 %4, i32 %1, i32 1
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}


define i32 @select_const_i32_from_fcmp(ptr nocapture readonly %v, i64 %n) {
; CHECK-LABEL: @select_const_i32_from_fcmp
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <4 x float>
; CHECK-VF4IC1-NEXT:   [[VEC_FCMP:%.*]] = fcmp ueq <4 x float> [[VEC_LOAD]], <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
; CHECK-VF4IC1-NEXT:   [[NOT:%.*]] = xor <4 x i1> [[VEC_FCMP]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 1, i32 2
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ 2, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds float, ptr %v, i64 %0
  %3 = load float, ptr %2, align 4
  %4 = fcmp ueq float %3, 3.0
  %5 = select i1 %4, i32 %1, i32 1
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}


define i32 @select_i32_from_icmp_same_inputs(i32 %a, i32 %b, i64 %n) {
; CHECK-LABEL: @select_i32_from_icmp_same_inputs
; CHECK-VF4IC1:      vector.ph:
; CHECK-VF4IC1:        [[TMP1:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
; CHECK-VF4IC1-NEXT:   [[SPLAT_OF_A:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-VF4IC1-NOT:   [[TMP2:%.*]] = insertelement <4 x i32> poison, i32 %b, i64 0
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_ICMP:%.*]] = icmp eq <4 x i32> [[SPLAT_OF_A]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC1-NEXT:   [[NOT:%.*]] = xor <4 x i1> [[VEC_ICMP]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 %b, i32 %a
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %4, %for.body ]
  %1 = phi i32 [ %a, %entry ], [ %3, %for.body ]
  %2 = icmp eq i32 %1, 3
  %3 = select i1 %2, i32 %1, i32 %b
  %4 = add nuw nsw i64 %0, 1
  %5 = icmp eq i64 %4, %n
  br i1 %5, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %3
}


define i32 @select_const_i32_from_icmp_mul_use(ptr nocapture readonly %v1, ptr %v2, i64 %n) {
; CHECK-VF4IC1-LABEL: define i32 @select_const_i32_from_icmp_mul_use(
; CHECK-VF4IC1-SAME: ptr nocapture readonly [[V1:%.*]], ptr [[V2:%.*]], i64 [[N:%.*]]) {
; CHECK-VF4IC1-NEXT:  [[ENTRY:.*]]:
; CHECK-VF4IC1-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N]], 4
; CHECK-VF4IC1-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %[[SCALAR_PH:.*]], label %[[VECTOR_PH:.*]]
; CHECK-VF4IC1:       [[VECTOR_PH]]:
; CHECK-VF4IC1-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N]], 4
; CHECK-VF4IC1-NEXT:    [[N_VEC:%.*]] = sub i64 [[N]], [[N_MOD_VF]]
; CHECK-VF4IC1-NEXT:    br label %[[VECTOR_BODY:.*]]
; CHECK-VF4IC1:       [[VECTOR_BODY]]:
; CHECK-VF4IC1-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %[[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF4IC1-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %[[VECTOR_PH]] ], [ [[TMP5:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF4IC1-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i32> [ <i32 poison, i32 poison, i32 poison, i32 0>, %[[VECTOR_PH]] ], [ [[TMP6:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF4IC1-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-VF4IC1-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, ptr [[V1]], i64 [[TMP0]]
; CHECK-VF4IC1-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i32 0
; CHECK-VF4IC1-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x i32>, ptr [[TMP2]], align 4
; CHECK-VF4IC1-NEXT:    [[TMP3:%.*]] = icmp eq <4 x i32> [[WIDE_LOAD]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC1-NEXT:    [[TMP4:%.*]] = xor <4 x i1> [[TMP3]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC1-NEXT:    [[TMP5]] = or <4 x i1> [[VEC_PHI]], [[TMP4]]
; CHECK-VF4IC1-NEXT:    [[TMP6]] = zext <4 x i1> [[TMP3]] to <4 x i32>
; CHECK-VF4IC1-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-VF4IC1-NEXT:    [[TMP7:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-VF4IC1-NEXT:    br i1 [[TMP7]], label %[[MIDDLE_BLOCK:.*]], label %[[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK-VF4IC1:       [[MIDDLE_BLOCK]]:
; CHECK-VF4IC1-NEXT:    [[TMP8:%.*]] = extractelement <4 x i32> [[TMP6]], i32 3
; CHECK-VF4IC1-NEXT:    [[TMP9:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[TMP5]])
; CHECK-VF4IC1-NEXT:    [[TMP10:%.*]] = freeze i1 [[TMP9]]
; CHECK-VF4IC1-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP10]], i32 7, i32 3
; CHECK-VF4IC1-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x i32> [[TMP6]], i32 3
; CHECK-VF4IC1-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N]], [[N_VEC]]
; CHECK-VF4IC1-NEXT:    br i1 [[CMP_N]], label %[[EXIT:.*]], label %[[SCALAR_PH]]
; CHECK-VF4IC1:       [[SCALAR_PH]]:
; CHECK-VF4IC1-NEXT:    [[SCALAR_RECUR_INIT:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[VECTOR_RECUR_EXTRACT]], %[[MIDDLE_BLOCK]] ]
; CHECK-VF4IC1-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], %[[MIDDLE_BLOCK]] ], [ 0, %[[ENTRY]] ]
; CHECK-VF4IC1-NEXT:    [[BC_MERGE_RDX:%.*]] = phi i32 [ 3, %[[ENTRY]] ], [ [[RDX_SELECT]], %[[MIDDLE_BLOCK]] ]
; CHECK-VF4IC1-NEXT:    br label %[[FOR_BODY:.*]]
; CHECK-VF4IC1:       [[FOR_BODY]]:
; CHECK-VF4IC1-NEXT:    [[TMP11:%.*]] = phi i64 [ [[BC_RESUME_VAL]], %[[SCALAR_PH]] ], [ [[TMP18:%.*]], %[[FOR_BODY]] ]
; CHECK-VF4IC1-NEXT:    [[TMP12:%.*]] = phi i32 [ [[BC_MERGE_RDX]], %[[SCALAR_PH]] ], [ [[TMP16:%.*]], %[[FOR_BODY]] ]
; CHECK-VF4IC1-NEXT:    [[SCALAR_RECUR:%.*]] = phi i32 [ [[SCALAR_RECUR_INIT]], %[[SCALAR_PH]] ], [ [[TMP17:%.*]], %[[FOR_BODY]] ]
; CHECK-VF4IC1-NEXT:    [[TMP13:%.*]] = getelementptr inbounds i32, ptr [[V1]], i64 [[TMP11]]
; CHECK-VF4IC1-NEXT:    [[TMP14:%.*]] = load i32, ptr [[TMP13]], align 4
; CHECK-VF4IC1-NEXT:    [[TMP15:%.*]] = icmp eq i32 [[TMP14]], 3
; CHECK-VF4IC1-NEXT:    [[TMP16]] = select i1 [[TMP15]], i32 [[TMP12]], i32 7
; CHECK-VF4IC1-NEXT:    [[TMP17]] = zext i1 [[TMP15]] to i32
; CHECK-VF4IC1-NEXT:    [[TMP18]] = add nuw nsw i64 [[TMP11]], 1
; CHECK-VF4IC1-NEXT:    [[TMP19:%.*]] = icmp eq i64 [[TMP18]], [[N]]
; CHECK-VF4IC1-NEXT:    br i1 [[TMP19]], label %[[EXIT]], label %[[FOR_BODY]], !llvm.loop [[LOOP3:![0-9]+]]
; CHECK-VF4IC1:       [[EXIT]]:
; CHECK-VF4IC1-NEXT:    [[DOTLCSSA1:%.*]] = phi i32 [ [[TMP16]], %[[FOR_BODY]] ], [ [[RDX_SELECT]], %[[MIDDLE_BLOCK]] ]
; CHECK-VF4IC1-NEXT:    [[DOTLCSSA:%.*]] = phi i32 [ [[TMP17]], %[[FOR_BODY]] ], [ [[TMP8]], %[[MIDDLE_BLOCK]] ]
; CHECK-VF4IC1-NEXT:    store i32 [[DOTLCSSA]], ptr [[V2]], align 4
; CHECK-VF4IC1-NEXT:    ret i32 [[DOTLCSSA1]]
;
; CHECK-VF4IC4-LABEL: define i32 @select_const_i32_from_icmp_mul_use(
; CHECK-VF4IC4-SAME: ptr nocapture readonly [[V1:%.*]], ptr [[V2:%.*]], i64 [[N:%.*]]) {
; CHECK-VF4IC4-NEXT:  [[ENTRY:.*]]:
; CHECK-VF4IC4-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N]], 16
; CHECK-VF4IC4-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %[[SCALAR_PH:.*]], label %[[VECTOR_PH:.*]]
; CHECK-VF4IC4:       [[VECTOR_PH]]:
; CHECK-VF4IC4-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N]], 16
; CHECK-VF4IC4-NEXT:    [[N_VEC:%.*]] = sub i64 [[N]], [[N_MOD_VF]]
; CHECK-VF4IC4-NEXT:    br label %[[VECTOR_BODY:.*]]
; CHECK-VF4IC4:       [[VECTOR_BODY]]:
; CHECK-VF4IC4-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %[[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF4IC4-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %[[VECTOR_PH]] ], [ [[TMP20:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF4IC4-NEXT:    [[VEC_PHI2:%.*]] = phi <4 x i1> [ zeroinitializer, %[[VECTOR_PH]] ], [ [[TMP21:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF4IC4-NEXT:    [[VEC_PHI3:%.*]] = phi <4 x i1> [ zeroinitializer, %[[VECTOR_PH]] ], [ [[TMP22:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF4IC4-NEXT:    [[VEC_PHI4:%.*]] = phi <4 x i1> [ zeroinitializer, %[[VECTOR_PH]] ], [ [[TMP23:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF4IC4-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i32> [ <i32 poison, i32 poison, i32 poison, i32 0>, %[[VECTOR_PH]] ], [ [[TMP27:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF4IC4-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-VF4IC4-NEXT:    [[TMP1:%.*]] = add i64 [[INDEX]], 4
; CHECK-VF4IC4-NEXT:    [[TMP2:%.*]] = add i64 [[INDEX]], 8
; CHECK-VF4IC4-NEXT:    [[TMP3:%.*]] = add i64 [[INDEX]], 12
; CHECK-VF4IC4-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, ptr [[V1]], i64 [[TMP0]]
; CHECK-VF4IC4-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, ptr [[V1]], i64 [[TMP1]]
; CHECK-VF4IC4-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[V1]], i64 [[TMP2]]
; CHECK-VF4IC4-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, ptr [[V1]], i64 [[TMP3]]
; CHECK-VF4IC4-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, ptr [[TMP4]], i32 0
; CHECK-VF4IC4-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i32, ptr [[TMP4]], i32 4
; CHECK-VF4IC4-NEXT:    [[TMP10:%.*]] = getelementptr inbounds i32, ptr [[TMP4]], i32 8
; CHECK-VF4IC4-NEXT:    [[TMP11:%.*]] = getelementptr inbounds i32, ptr [[TMP4]], i32 12
; CHECK-VF4IC4-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x i32>, ptr [[TMP8]], align 4
; CHECK-VF4IC4-NEXT:    [[WIDE_LOAD5:%.*]] = load <4 x i32>, ptr [[TMP9]], align 4
; CHECK-VF4IC4-NEXT:    [[WIDE_LOAD6:%.*]] = load <4 x i32>, ptr [[TMP10]], align 4
; CHECK-VF4IC4-NEXT:    [[WIDE_LOAD7:%.*]] = load <4 x i32>, ptr [[TMP11]], align 4
; CHECK-VF4IC4-NEXT:    [[TMP12:%.*]] = icmp eq <4 x i32> [[WIDE_LOAD]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC4-NEXT:    [[TMP13:%.*]] = icmp eq <4 x i32> [[WIDE_LOAD5]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC4-NEXT:    [[TMP14:%.*]] = icmp eq <4 x i32> [[WIDE_LOAD6]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC4-NEXT:    [[TMP15:%.*]] = icmp eq <4 x i32> [[WIDE_LOAD7]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC4-NEXT:    [[TMP16:%.*]] = xor <4 x i1> [[TMP12]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC4-NEXT:    [[TMP17:%.*]] = xor <4 x i1> [[TMP13]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC4-NEXT:    [[TMP18:%.*]] = xor <4 x i1> [[TMP14]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC4-NEXT:    [[TMP19:%.*]] = xor <4 x i1> [[TMP15]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC4-NEXT:    [[TMP20]] = or <4 x i1> [[VEC_PHI]], [[TMP16]]
; CHECK-VF4IC4-NEXT:    [[TMP21]] = or <4 x i1> [[VEC_PHI2]], [[TMP17]]
; CHECK-VF4IC4-NEXT:    [[TMP22]] = or <4 x i1> [[VEC_PHI3]], [[TMP18]]
; CHECK-VF4IC4-NEXT:    [[TMP23]] = or <4 x i1> [[VEC_PHI4]], [[TMP19]]
; CHECK-VF4IC4-NEXT:    [[TMP24:%.*]] = zext <4 x i1> [[TMP12]] to <4 x i32>
; CHECK-VF4IC4-NEXT:    [[TMP25:%.*]] = zext <4 x i1> [[TMP13]] to <4 x i32>
; CHECK-VF4IC4-NEXT:    [[TMP26:%.*]] = zext <4 x i1> [[TMP14]] to <4 x i32>
; CHECK-VF4IC4-NEXT:    [[TMP27]] = zext <4 x i1> [[TMP15]] to <4 x i32>
; CHECK-VF4IC4-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 16
; CHECK-VF4IC4-NEXT:    [[TMP28:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-VF4IC4-NEXT:    br i1 [[TMP28]], label %[[MIDDLE_BLOCK:.*]], label %[[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK-VF4IC4:       [[MIDDLE_BLOCK]]:
; CHECK-VF4IC4-NEXT:    [[TMP29:%.*]] = extractelement <4 x i32> [[TMP27]], i32 3
; CHECK-VF4IC4-NEXT:    [[BIN_RDX:%.*]] = or <4 x i1> [[TMP21]], [[TMP20]]
; CHECK-VF4IC4-NEXT:    [[BIN_RDX8:%.*]] = or <4 x i1> [[TMP22]], [[BIN_RDX]]
; CHECK-VF4IC4-NEXT:    [[BIN_RDX9:%.*]] = or <4 x i1> [[TMP23]], [[BIN_RDX8]]
; CHECK-VF4IC4-NEXT:    [[TMP30:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[BIN_RDX9]])
; CHECK-VF4IC4-NEXT:    [[TMP31:%.*]] = freeze i1 [[TMP30]]
; CHECK-VF4IC4-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP31]], i32 7, i32 3
; CHECK-VF4IC4-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x i32> [[TMP27]], i32 3
; CHECK-VF4IC4-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N]], [[N_VEC]]
; CHECK-VF4IC4-NEXT:    br i1 [[CMP_N]], label %[[EXIT:.*]], label %[[SCALAR_PH]]
; CHECK-VF4IC4:       [[SCALAR_PH]]:
; CHECK-VF4IC4-NEXT:    [[SCALAR_RECUR_INIT:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[VECTOR_RECUR_EXTRACT]], %[[MIDDLE_BLOCK]] ]
; CHECK-VF4IC4-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], %[[MIDDLE_BLOCK]] ], [ 0, %[[ENTRY]] ]
; CHECK-VF4IC4-NEXT:    [[BC_MERGE_RDX:%.*]] = phi i32 [ 3, %[[ENTRY]] ], [ [[RDX_SELECT]], %[[MIDDLE_BLOCK]] ]
; CHECK-VF4IC4-NEXT:    br label %[[FOR_BODY:.*]]
; CHECK-VF4IC4:       [[FOR_BODY]]:
; CHECK-VF4IC4-NEXT:    [[TMP32:%.*]] = phi i64 [ [[BC_RESUME_VAL]], %[[SCALAR_PH]] ], [ [[TMP39:%.*]], %[[FOR_BODY]] ]
; CHECK-VF4IC4-NEXT:    [[TMP33:%.*]] = phi i32 [ [[BC_MERGE_RDX]], %[[SCALAR_PH]] ], [ [[TMP37:%.*]], %[[FOR_BODY]] ]
; CHECK-VF4IC4-NEXT:    [[SCALAR_RECUR:%.*]] = phi i32 [ [[SCALAR_RECUR_INIT]], %[[SCALAR_PH]] ], [ [[TMP38:%.*]], %[[FOR_BODY]] ]
; CHECK-VF4IC4-NEXT:    [[TMP34:%.*]] = getelementptr inbounds i32, ptr [[V1]], i64 [[TMP32]]
; CHECK-VF4IC4-NEXT:    [[TMP35:%.*]] = load i32, ptr [[TMP34]], align 4
; CHECK-VF4IC4-NEXT:    [[TMP36:%.*]] = icmp eq i32 [[TMP35]], 3
; CHECK-VF4IC4-NEXT:    [[TMP37]] = select i1 [[TMP36]], i32 [[TMP33]], i32 7
; CHECK-VF4IC4-NEXT:    [[TMP38]] = zext i1 [[TMP36]] to i32
; CHECK-VF4IC4-NEXT:    [[TMP39]] = add nuw nsw i64 [[TMP32]], 1
; CHECK-VF4IC4-NEXT:    [[TMP40:%.*]] = icmp eq i64 [[TMP39]], [[N]]
; CHECK-VF4IC4-NEXT:    br i1 [[TMP40]], label %[[EXIT]], label %[[FOR_BODY]], !llvm.loop [[LOOP3:![0-9]+]]
; CHECK-VF4IC4:       [[EXIT]]:
; CHECK-VF4IC4-NEXT:    [[DOTLCSSA1:%.*]] = phi i32 [ [[TMP37]], %[[FOR_BODY]] ], [ [[RDX_SELECT]], %[[MIDDLE_BLOCK]] ]
; CHECK-VF4IC4-NEXT:    [[DOTLCSSA:%.*]] = phi i32 [ [[TMP38]], %[[FOR_BODY]] ], [ [[TMP29]], %[[MIDDLE_BLOCK]] ]
; CHECK-VF4IC4-NEXT:    store i32 [[DOTLCSSA]], ptr [[V2]], align 4
; CHECK-VF4IC4-NEXT:    ret i32 [[DOTLCSSA1]]
;
; CHECK-VF1IC4-LABEL: define i32 @select_const_i32_from_icmp_mul_use(
; CHECK-VF1IC4-SAME: ptr nocapture readonly [[V1:%.*]], ptr [[V2:%.*]], i64 [[N:%.*]]) {
; CHECK-VF1IC4-NEXT:  [[ENTRY:.*]]:
; CHECK-VF1IC4-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N]], 4
; CHECK-VF1IC4-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %[[SCALAR_PH:.*]], label %[[VECTOR_PH:.*]]
; CHECK-VF1IC4:       [[VECTOR_PH]]:
; CHECK-VF1IC4-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N]], 4
; CHECK-VF1IC4-NEXT:    [[N_VEC:%.*]] = sub i64 [[N]], [[N_MOD_VF]]
; CHECK-VF1IC4-NEXT:    br label %[[VECTOR_BODY:.*]]
; CHECK-VF1IC4:       [[VECTOR_BODY]]:
; CHECK-VF1IC4-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %[[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF1IC4-NEXT:    [[VEC_PHI:%.*]] = phi i1 [ false, %[[VECTOR_PH]] ], [ [[TMP20:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF1IC4-NEXT:    [[VEC_PHI2:%.*]] = phi i1 [ false, %[[VECTOR_PH]] ], [ [[TMP21:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF1IC4-NEXT:    [[VEC_PHI3:%.*]] = phi i1 [ false, %[[VECTOR_PH]] ], [ [[TMP22:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF1IC4-NEXT:    [[VEC_PHI4:%.*]] = phi i1 [ false, %[[VECTOR_PH]] ], [ [[TMP23:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF1IC4-NEXT:    [[VECTOR_RECUR:%.*]] = phi i32 [ 0, %[[VECTOR_PH]] ], [ [[VECTOR_RECUR_EXTRACT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-VF1IC4-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-VF1IC4-NEXT:    [[TMP1:%.*]] = add i64 [[INDEX]], 1
; CHECK-VF1IC4-NEXT:    [[TMP2:%.*]] = add i64 [[INDEX]], 2
; CHECK-VF1IC4-NEXT:    [[TMP3:%.*]] = add i64 [[INDEX]], 3
; CHECK-VF1IC4-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, ptr [[V1]], i64 [[TMP0]]
; CHECK-VF1IC4-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, ptr [[V1]], i64 [[TMP1]]
; CHECK-VF1IC4-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[V1]], i64 [[TMP2]]
; CHECK-VF1IC4-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, ptr [[V1]], i64 [[TMP3]]
; CHECK-VF1IC4-NEXT:    [[TMP8:%.*]] = load i32, ptr [[TMP4]], align 4
; CHECK-VF1IC4-NEXT:    [[TMP9:%.*]] = load i32, ptr [[TMP5]], align 4
; CHECK-VF1IC4-NEXT:    [[TMP10:%.*]] = load i32, ptr [[TMP6]], align 4
; CHECK-VF1IC4-NEXT:    [[TMP11:%.*]] = load i32, ptr [[TMP7]], align 4
; CHECK-VF1IC4-NEXT:    [[TMP12:%.*]] = icmp eq i32 [[TMP8]], 3
; CHECK-VF1IC4-NEXT:    [[TMP13:%.*]] = icmp eq i32 [[TMP9]], 3
; CHECK-VF1IC4-NEXT:    [[TMP14:%.*]] = icmp eq i32 [[TMP10]], 3
; CHECK-VF1IC4-NEXT:    [[TMP15:%.*]] = icmp eq i32 [[TMP11]], 3
; CHECK-VF1IC4-NEXT:    [[TMP16:%.*]] = xor i1 [[TMP12]], true
; CHECK-VF1IC4-NEXT:    [[TMP17:%.*]] = xor i1 [[TMP13]], true
; CHECK-VF1IC4-NEXT:    [[TMP18:%.*]] = xor i1 [[TMP14]], true
; CHECK-VF1IC4-NEXT:    [[TMP19:%.*]] = xor i1 [[TMP15]], true
; CHECK-VF1IC4-NEXT:    [[TMP20]] = or i1 [[VEC_PHI]], [[TMP16]]
; CHECK-VF1IC4-NEXT:    [[TMP21]] = or i1 [[VEC_PHI2]], [[TMP17]]
; CHECK-VF1IC4-NEXT:    [[TMP22]] = or i1 [[VEC_PHI3]], [[TMP18]]
; CHECK-VF1IC4-NEXT:    [[TMP23]] = or i1 [[VEC_PHI4]], [[TMP19]]
; CHECK-VF1IC4-NEXT:    [[TMP24:%.*]] = zext i1 [[TMP12]] to i32
; CHECK-VF1IC4-NEXT:    [[TMP25:%.*]] = zext i1 [[TMP13]] to i32
; CHECK-VF1IC4-NEXT:    [[TMP26:%.*]] = zext i1 [[TMP14]] to i32
; CHECK-VF1IC4-NEXT:    [[VECTOR_RECUR_EXTRACT]] = zext i1 [[TMP15]] to i32
; CHECK-VF1IC4-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-VF1IC4-NEXT:    [[TMP27:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-VF1IC4-NEXT:    br i1 [[TMP27]], label %[[MIDDLE_BLOCK:.*]], label %[[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK-VF1IC4:       [[MIDDLE_BLOCK]]:
; CHECK-VF1IC4-NEXT:    [[BIN_RDX:%.*]] = or i1 [[TMP21]], [[TMP20]]
; CHECK-VF1IC4-NEXT:    [[BIN_RDX5:%.*]] = or i1 [[TMP22]], [[BIN_RDX]]
; CHECK-VF1IC4-NEXT:    [[BIN_RDX6:%.*]] = or i1 [[TMP23]], [[BIN_RDX5]]
; CHECK-VF1IC4-NEXT:    [[TMP28:%.*]] = freeze i1 [[BIN_RDX6]]
; CHECK-VF1IC4-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP28]], i32 7, i32 3
; CHECK-VF1IC4-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N]], [[N_VEC]]
; CHECK-VF1IC4-NEXT:    br i1 [[CMP_N]], label %[[EXIT:.*]], label %[[SCALAR_PH]]
; CHECK-VF1IC4:       [[SCALAR_PH]]:
; CHECK-VF1IC4-NEXT:    [[SCALAR_RECUR_INIT:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[VECTOR_RECUR_EXTRACT]], %[[MIDDLE_BLOCK]] ]
; CHECK-VF1IC4-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], %[[MIDDLE_BLOCK]] ], [ 0, %[[ENTRY]] ]
; CHECK-VF1IC4-NEXT:    [[BC_MERGE_RDX:%.*]] = phi i32 [ 3, %[[ENTRY]] ], [ [[RDX_SELECT]], %[[MIDDLE_BLOCK]] ]
; CHECK-VF1IC4-NEXT:    br label %[[FOR_BODY:.*]]
; CHECK-VF1IC4:       [[FOR_BODY]]:
; CHECK-VF1IC4-NEXT:    [[TMP29:%.*]] = phi i64 [ [[BC_RESUME_VAL]], %[[SCALAR_PH]] ], [ [[TMP36:%.*]], %[[FOR_BODY]] ]
; CHECK-VF1IC4-NEXT:    [[TMP30:%.*]] = phi i32 [ [[BC_MERGE_RDX]], %[[SCALAR_PH]] ], [ [[TMP34:%.*]], %[[FOR_BODY]] ]
; CHECK-VF1IC4-NEXT:    [[SCALAR_RECUR:%.*]] = phi i32 [ [[SCALAR_RECUR_INIT]], %[[SCALAR_PH]] ], [ [[TMP35:%.*]], %[[FOR_BODY]] ]
; CHECK-VF1IC4-NEXT:    [[TMP31:%.*]] = getelementptr inbounds i32, ptr [[V1]], i64 [[TMP29]]
; CHECK-VF1IC4-NEXT:    [[TMP32:%.*]] = load i32, ptr [[TMP31]], align 4
; CHECK-VF1IC4-NEXT:    [[TMP33:%.*]] = icmp eq i32 [[TMP32]], 3
; CHECK-VF1IC4-NEXT:    [[TMP34]] = select i1 [[TMP33]], i32 [[TMP30]], i32 7
; CHECK-VF1IC4-NEXT:    [[TMP35]] = zext i1 [[TMP33]] to i32
; CHECK-VF1IC4-NEXT:    [[TMP36]] = add nuw nsw i64 [[TMP29]], 1
; CHECK-VF1IC4-NEXT:    [[TMP37:%.*]] = icmp eq i64 [[TMP36]], [[N]]
; CHECK-VF1IC4-NEXT:    br i1 [[TMP37]], label %[[EXIT]], label %[[FOR_BODY]], !llvm.loop [[LOOP3:![0-9]+]]
; CHECK-VF1IC4:       [[EXIT]]:
; CHECK-VF1IC4-NEXT:    [[DOTLCSSA1:%.*]] = phi i32 [ [[TMP34]], %[[FOR_BODY]] ], [ [[RDX_SELECT]], %[[MIDDLE_BLOCK]] ]
; CHECK-VF1IC4-NEXT:    [[DOTLCSSA:%.*]] = phi i32 [ [[TMP35]], %[[FOR_BODY]] ], [ [[VECTOR_RECUR_EXTRACT]], %[[MIDDLE_BLOCK]] ]
; CHECK-VF1IC4-NEXT:    store i32 [[DOTLCSSA]], ptr [[V2]], align 4
; CHECK-VF1IC4-NEXT:    ret i32 [[DOTLCSSA1]]
;
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %8, %for.body ]
  %1 = phi i32 [ 3, %entry ], [ %6, %for.body ]
  %2 = phi i32 [ 0, %entry ], [ %7, %for.body ]
  %3 = getelementptr inbounds i32, ptr %v1, i64 %0
  %4 = load i32, ptr %3, align 4
  %5 = icmp eq i32 %4, 3
  %6 = select i1 %5, i32 %1, i32 7
  %7 = zext i1 %5 to i32
  %8 = add nuw nsw i64 %0, 1
  %9 = icmp eq i64 %8, %n
  br i1 %9, label %exit, label %for.body

exit:                                     ; preds = %for.body
  store i32 %7, ptr %v2, align 4
  ret i32 %6
}

; Negative tests

; We don't support FP reduction variables at the moment.
define float @select_const_f32_from_icmp(ptr nocapture readonly %v, i64 %n) {
; CHECK: @select_const_f32_from_icmp
; CHECK-NOT: vector.body
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi fast float [ 3.0, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select fast i1 %4, float %1, float 7.0
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret float %5
}


; We don't support selecting loop-variant values.
define i32 @select_variant_i32_from_icmp(ptr nocapture readonly %v1, ptr nocapture readonly %v2, i64 %n) {
; CHECK-LABEL: @select_variant_i32_from_icmp
; CHECK-NOT: vector.body
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %8, %for.body ]
  %1 = phi i32 [ 3, %entry ], [ %7, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v1, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = getelementptr inbounds i32, ptr %v2, i64 %0
  %5 = load i32, ptr %4, align 4
  %6 = icmp eq i32 %3, 3
  %7 = select i1 %6, i32 %1, i32 %5
  %8 = add nuw nsw i64 %0, 1
  %9 = icmp eq i64 %8, %n
  br i1 %9, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %7
}


; We only support selects where the input comes from the same PHI as the
; reduction PHI. In the example below, the select uses the induction
; variable input and the icmp uses the reduction PHI.
define i32 @select_i32_from_icmp_non_redux_phi(i32 %a, i32 %b, i32 %n) {
; CHECK-LABEL: @select_i32_from_icmp_non_redux_phi
; CHECK-NOT: vector.body
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i32 [ 0, %entry ], [ %4, %for.body ]
  %1 = phi i32 [ %a, %entry ], [ %3, %for.body ]
  %2 = icmp eq i32 %1, 3
  %3 = select i1 %2, i32 %0, i32 %b
  %4 = add nuw nsw i32 %0, 1
  %5 = icmp eq i32 %4, %n
  br i1 %5, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %3
}
