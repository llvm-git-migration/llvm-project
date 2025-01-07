func.func @gemm_no_scf_sm100_1cta(%133 : i32, %arg1: i32, %c127_i32:i32) {
    %134 = llvm.bitcast %133 : i32 to f32        
    gpu.printf "]"
    %135 = arith.cmpi slt, %arg1, %c127_i32 : i32
    scf.if %135 {
        gpu.printf ", "
    } 

    %0 = gpu.thread_id x
    gpu.printf "Hello from %d\n", %0 : index
    func.return
}
