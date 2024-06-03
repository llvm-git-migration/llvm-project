;; Tests memprof with collecting AccessCountHistograms

;; Avoid failures on big-endian systems that can't read the profile properly
; REQUIRES: x86_64-linux

;; TODO: Use text profile inputs once that is available for memprof.
;; # To update the Inputs below, run Inputs/update_memprof_inputs.sh.
;; # To generate below LLVM IR for use in matching:
;; $ clang++ -gmlt -fdebug-info-for-profiling -fno-omit-frame-pointer \
;;	-fno-optimize-sibling-calls memprof.cc -S -emit-llvm


;; # To update the Inputs below, run Inputs/update_memprof_inputs.sh.



; RUN: env MEMPROF_OPTIONS=print_text=true:log_path=stdout:histogram=true %S/Inputs/memprof_histogram.exe 2>&1 | FileCheck %s --check-prefix HistogramText


; HistogramText: AcccessCountHistogram
; HistogramText: AcccessCountHistogram[8]: 1 1 1 1 1 1 1 1 
; HistogramText: AcccessCountHistogram[8]: 8 7 6 5 4 3 2 1 
; HistogramText: AcccessCountHistogram[8]: 1 0 0 0 0 0 0 0 
; HistogramText: AcccessCountHistogram[8]: 21 21 21 21 21 21 21 21 
; HistogramText: AcccessCountHistogram[8]: 168 147 126 105 84 63 42 21


; RUN: llvm-profdata  show %S/Inputs/memprof_histogram.memprofraw --profiled-binary=%S/Inputs/memprof_histogram.exe --memory 2>&1 | FileCheck %s  --check-prefix HistogramYaml

;HistogramYaml: AccessHistogramValues: -8 -7 -6 -5 -4 -3 -2 -1
;HistogramYaml: AccessHistogramValues: -1 -0 -0 -0 -0 -0 -0 -0
;HistogramYaml: AccessHistogramValues: -21 -21 -21 -21 -21 -21 -21 -21
;HistogramYaml: AccessHistogramValues: -168 -147 -126 -105 -84 -63 -42 -21
