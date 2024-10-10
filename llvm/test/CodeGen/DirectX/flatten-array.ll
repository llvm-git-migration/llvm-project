; RUN: opt -S -dxil-flatten-arrays %s | FileCheck %s

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
    ; CHECK-NOT: getelementptr inbounds [3 x [3 x i32]], ptr %1, i32 0, i32 {{.*}}, i32 {{.*}}
    ; CHECK-NOT: getelementptr inbounds [3 x i32], [3 x i32]* {{.*}}, i32 {{.*}}, i32 {{.*}}
    %1 = alloca [3 x [3 x i32]], align 4
    %g2d0 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* %1, i32 0, i32 0
    %g1d0_1 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d0, i32 0, i32 0
    %g1d0_2 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d0, i32 0, i32 1
    %g1d0_3 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d0, i32 0, i32 2
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
    %1 = alloca [2 x[2 x [2 x i32]]], align 4
    %g3d0 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %1, i32 0, i32 0
    %g2d0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0, i32 0, i32 0
    %g1d0_1 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0, i32 0, i32 0
    %g1d0_2 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0, i32 0, i32 1
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
    %1 = alloca [2x[2 x[2 x [2 x i32]]]], align 4
    ret void
}