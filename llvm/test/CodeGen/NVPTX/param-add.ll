; RUN: llc < %s -march=nvptx64 --debug-counter=dagcombine=0 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 | %ptxas-verify %}

%struct.8float = type <{ [8 x float] }>

declare i32 @callee(%struct.8float %a)

define i32 @test(%struct.8float alignstack(32) %data) {
  ;CHECK-NOT: add.
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+1];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+2];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+3];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+4];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+5];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+6];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+7];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+8];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+9];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+10];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+11];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+12];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+13];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+14];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+15];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+16];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+17];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+18];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+19];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+20];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+21];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+22];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+23];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+24];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+26];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+27];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+28];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+29];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+30];
  ;CHECK-DAG: ld.param.u8 %r{{.*}}, [test_param_0+31];

  %1 = call i32 @callee(%struct.8float %data)
  ret i32 %1
}
