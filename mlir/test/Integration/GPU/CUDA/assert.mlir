// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-format=%gpu_compilation_format" \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void 2>&1 \
// RUN: | FileCheck %s

// CHECK-DAG: thread 0: print after passing assertion
// CHECK-DAG: thread 1: print after passing assertion
// CHECK-DAG: mlir/test/Integration/GPU/CUDA/assert.mlir:{{.*}}: (unknown): block: [0,0,0], thread: [0,0,0] Assertion `failing assertion` failed.
// CHECK-DAG: mlir/test/Integration/GPU/CUDA/assert.mlir:{{.*}}: (unknown): block: [0,0,0], thread: [1,0,0] Assertion `failing assertion` failed.
// CHECK-NOT: print after failing assertion

module attributes {gpu.container_module} {
gpu.module @kernels {
gpu.func @hello(%c0: i1, %c1: i1) kernel {
  %0 = gpu.thread_id x
  gpu.assert %c1, "passing assertion"
  gpu.printf "thread %lld: print after passing assertion\n" %0 : index
  gpu.assert %c0, "failing assertion"
  gpu.printf "thread %lld: print after failing assertion\n" %0 : index
  gpu.return
}
}

func.func @main() {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0_i1 = arith.constant 0 : i1
  %c1_i1 = arith.constant 1 : i1
  gpu.launch_func @kernels::@hello
      blocks in (%c1, %c1, %c1)
      threads in (%c2, %c1, %c1)
      args(%c0_i1 : i1, %c1_i1 : i1)
  return
}
}
