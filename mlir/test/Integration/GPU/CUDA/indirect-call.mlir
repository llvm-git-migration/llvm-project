// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-format=%gpu_compilation_format" \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// CHECK: Hello from 0, 1, 3.000000
module attributes {gpu.container_module} {
    gpu.module @kernels {
        func.func @hello(%arg0 : f32) {
            %0 = gpu.thread_id x
            %csti8 = arith.constant 2 : i8
            gpu.printf "Hello from %lld, %d, %f\n" %0, %csti8, %arg0  : index, i8, f32
            return
        }
    
        gpu.func @hello_indirect() kernel {
            %cstf32 = arith.constant 3.0 : f32
            %func_ref = func.constant @hello : (f32) -> ()
            func.call_indirect %func_ref(%cstf32) : (f32) -> ()
            gpu.return
        }
    }

    func.func @main() {
        %c1 = arith.constant 1 : index
        gpu.launch_func @kernels::@hello_indirect
            blocks in (%c1, %c1, %c1)
            threads in (%c1, %c1, %c1)
        return
    }
}
