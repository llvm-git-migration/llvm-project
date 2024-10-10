; RUN: opt -S -dxil-flatten-arrays  %s | FileCheck %s

; CHECK-LABEL: alloca_2d_test
define void @alloca_2d_test ()  {
    ; CHECK: alloca [9 x i32], align 4
    ; CHECK-NOT: alloca [3 x [3 x i32]], align 4
    %1 = alloca [3 x [3 x i32]], align 4
    ret void
}

; CHECK-LABEL: alloca_3d_test
define void @alloca_3d_test ()  {
    ; CHECK: alloca [8 x i32], align 4
    ; CHECK-NOT: alloca [2 x[2 x [2 x i32]]], align 4
    %1 = alloca [2 x[2 x [2 x i32]]], align 4
    ret void
}

; CHECK-LABEL: alloca_4d_test
define void @alloca_4d_test ()  {
    ; CHECK: alloca [16 x i32], align 4
    ; CHECK-NOT: alloca [ 2x[2 x[2 x [2 x i32]]]], align 4
    %1 = alloca [2x[2 x[2 x [2 x i32]]]], align 4
    ret void
}

; CHECK-LABEL: gep_2d_test
define void @gep_2d_test ()  {
    ; CHECK: [[a:%.*]] = alloca [9 x i32], align 4
    ; CHECK-COUNT-9: getelementptr inbounds [9 x i32], ptr [[a]], i32 {{[0-8]}}
    ; CHECK-NOT: getelementptr inbounds [3 x [3 x i32]], ptr %1, i32 0, i32 0, i32 {{[0-2]}}
    ; CHECK-NOT: getelementptr inbounds [3 x i32], [3 x i32]* {{.*}}, i32 0, i32 {{[0-2]}}
    %1 = alloca [3 x [3 x i32]], align 4
    %g2d0 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* %1, i32 0, i32 0
    %g1d_1 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d0, i32 0, i32 0
    %g1d_2 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d0, i32 0, i32 1
    %g1d_3 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d0, i32 0, i32 2
    %g2d1 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* %1, i32 0, i32 1
    %g1d1_1 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d1, i32 0, i32 0
    %g1d1_2 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d1, i32 0, i32 1
    %g1d1_3 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d1, i32 0, i32 2
    %g2d2 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* %1, i32 0, i32 2
    %g1d2_1 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d2, i32 0, i32 0
    %g1d2_2 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d2, i32 0, i32 1
    %g1d2_3 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d2, i32 0, i32 2
    
    ret void
}

; CHECK-LABEL: gep_3d_test
define void @gep_3d_test ()  {
    ; CHECK: [[a:%.*]] = alloca [8 x i32], align 4
    ; CHECK-COUNT-8: getelementptr inbounds [8 x i32], ptr [[a]], i32 {{[0-7]}}
    ; CHECK-NOT: getelementptr inbounds [2 x[2 x [2 x i32]]], ptr %1, i32 0, i32 0, i32 {{[0-1]}}
    ; CHECK-NOT: getelementptr inbounds [2 x [2 x i32]], ptr {{.*}}, i32 0, i32 0, i32 {{[0-1]}}
    ; CHECK-NOT: getelementptr inbounds [2 x i32], [2 x i32]* {{.*}}, i32 0, i32 {{[0-1]}}
    %1 = alloca [2 x[2 x [2 x i32]]], align 4
    %g3d0 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %1, i32 0, i32 0
    %g2d0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0, i32 0, i32 0
    %g1d_1 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0, i32 0, i32 0
    %g1d_2 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0, i32 0, i32 1
    %g2d1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0, i32 0, i32 1
    %g1d1_1 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d1, i32 0, i32 0
    %g1d1_2 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d1, i32 0, i32 1
    %g3d1 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %1, i32 0, i32 1
    %g2d2 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d1, i32 0, i32 0
    %g1d2_1 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d2, i32 0, i32 0
    %g1d2_2 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d2, i32 0, i32 1
    %g2d3 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d1, i32 0, i32 1
    %g1d3_1 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d3, i32 0, i32 0
    %g1d3_2 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d3, i32 0, i32 1
    ret void
}

; CHECK-LABEL: gep_4d_test
define void @gep_4d_test ()  {
    ; CHECK: [[a:%.*]] = alloca [16 x i32], align 4
    ; CHECK-COUNT-16: getelementptr inbounds [16 x i32], ptr [[a]], i32 {{[0-9]|1[0-5]}}
    ; CHECK-NOT: getelementptr inbounds [2x[2 x[2 x [2 x i32]]]],, ptr %1, i32 0, i32 0, i32 {{[0-1]}}
    ; CHECK-NOT: getelementptr inbounds [2 x[2 x [2 x i32]]], ptr {{.*}}, i32 0, i32 0, i32 {{[0-1]}}
    ; CHECK-NOT: getelementptr inbounds [2 x [2 x i32]], ptr {{.*}}, i32 0, i32 0, i32 {{[0-1]}}
    ; CHECK-NOT: getelementptr inbounds [2 x i32], [2 x i32]* {{.*}}, i32 0, i32 {{[0-1]}}
    %1 = alloca [2x[2 x[2 x [2 x i32]]]], align 4
    %g4d0 = getelementptr inbounds [2x[2 x[2 x [2 x i32]]]], [2x[2 x[2 x [2 x i32]]]]* %1, i32 0, i32 0
    %g3d0 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %g4d0, i32 0, i32 0
    %g2d0_0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0, i32 0, i32 0
    %g1d_0 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_0, i32 0, i32 0
    %g1d_1 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_0, i32 0, i32 1
    %g2d0_1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0, i32 0, i32 1
    %g1d_2 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_1, i32 0, i32 0
    %g1d_3 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_1, i32 0, i32 1
    %g3d1 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %g4d0, i32 0, i32 1
    %g2d0_2 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d1, i32 0, i32 0
    %g1d_4 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_2, i32 0, i32 0
    %g1d_5 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_2, i32 0, i32 1
    %g2d1_2 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d1, i32 0, i32 1
    %g1d_6 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d1_2, i32 0, i32 0
    %g1d_7 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d1_2, i32 0, i32 1
    %g4d1 = getelementptr inbounds [2x[2 x[2 x [2 x i32]]]], [2x[2 x[2 x [2 x i32]]]]* %1, i32 0, i32 1
    %g3d0_1 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %g4d1, i32 0, i32 0
    %g2d0_3 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0_1, i32 0, i32 0
    %g1d_8 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_3, i32 0, i32 0
    %g1d_9 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_3, i32 0, i32 1
    %g2d0_4 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0_1, i32 0, i32 1
    %g1d_10 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_4, i32 0, i32 0
    %g1d_11 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_4, i32 0, i32 1
    %g3d1_1 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %g4d1, i32 0, i32 1
    %g2d0_5 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d1_1, i32 0, i32 0
    %g1d_12 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_5, i32 0, i32 0
    %g1d_13 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_5, i32 0, i32 1
    %g2d1_3 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d1_1, i32 0, i32 1
    %g1d_14 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d1_3, i32 0, i32 0
    %g1d_15 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d1_3, i32 0, i32 1
    ret void
}

; CHECK-LABEL: bitcast_2d_test
define void @bitcast_2d_test ()  {
    ; CHECK: alloca [9 x i32], align 4
    %1 = alloca [3 x [3 x i32]], align 4
    bitcast [3 x [3 x i32]]* %1 to i8*
    ret void
}

; CHECK-LABEL: bitcast_3d_test
define void @bitcast_3d_test ()  {
    ; CHECK: alloca [8 x i32], align 4
    %1 = alloca [2 x[2 x [2 x i32]]], align 4
    bitcast [2 x[2 x [2 x i32]]]* %1 to i8*
    ret void
}

; CHECK-LABEL: bitcast_4d_test
define void @bitcast_4d_test ()  {
    ; CHECK: alloca [16 x i32], align 4
    %1 = alloca [2x[2 x[2 x [2 x i32]]]], align 4
    bitcast [2x[2 x[2 x [2 x i32]]]]* %1 to i8*
    ret void
}