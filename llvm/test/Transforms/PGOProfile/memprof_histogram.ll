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


; HistogramText: access_count (ave/min/max): 8.00 / 8 / 8
; HistogramText: AcccessCountHistogram[8]: 1 1 1 1 1 1 1 1 
; HistogramText: access_count (ave/min/max): 36.00 / 36 / 36
; HistogramText: AcccessCountHistogram[8]: 8 7 6 5 4 3 2 1 
; HistogramText: access_count (ave/min/max): 1.00 / 1 / 1
; HistogramText: AcccessCountHistogram[8]: 1 0 0 0 0 0 0 0 
; HistogramText: access_count (ave/min/max): 8.00 / 8 / 8
; HistogramText: AcccessCountHistogram[8]: 21 21 21 21 21 21 21 21 
; HistogramText: access_count (ave/min/max): 36.00 / 36 / 36
; HistogramText: AcccessCountHistogram[8]: 168 147 126 105 84 63 42 21


; RUN: llvm-profdata show %S/Inputs/memprof_histogram.memprofraw --profiled-binary=%S/Inputs/memprof_histogram.exe --memory 2>&1 | FileCheck %s  --check-prefix HistogramYaml

; HistogramYaml: TotalAccessCount: 8
; HistogramYaml: AccessHistogramValues: -1 -1 -1 -1 -1 -1 -1 -1
; HistogramYaml: TotalAccessCount: 36
; HistogramYaml: AccessHistogramValues: -8 -7 -6 -5 -4 -3 -2 -1
; HistogramYaml: TotalAccessCount: 1
; HistogramYaml: AccessHistogramValues: -1 -0 -0 -0 -0 -0 -0 -0
; HistogramYaml: TotalAccessCount: 168
; HistogramYaml: AccessHistogramValues: -21 -21 -21 -21 -21 -21 -21 -21
; HistogramYaml: TotalAccessCount: 756
; HistogramYaml: AccessHistogramValues: -168 -147 -126 -105 -84 -63 -42 -21


; RUN: env MEMPROF_OPTIONS=print_text=true:log_path=stdout:histogram=true %S/Inputs/memprof_histogram_padding.exe 2>&1 | FileCheck %s --check-prefix HistogramPaddingText


; HistogramPaddingText: access_count (ave/min/max): 5.00 / 5 / 5
; HistogramPaddingText: AcccessCountHistogram[3]: 2 1 2
; HistogramPaddingText: access_count (ave/min/max): 4.00 / 4 / 4
; HistogramPaddingText: AcccessCountHistogram[6]: 2 0 0 0 1 1 


; RUN: llvm-profdata show %S/Inputs/memprof_histogram_padding.memprofraw --profiled-binary=%S/Inputs/memprof_histogram_padding.exe --memory 2>&1 | FileCheck %s --check-prefix HistogramPaddingYaml

; HistogramPaddingYaml: TotalAccessCount: 5
; HistogramPaddingYaml: AccessHistogramValues: -2 -1 -2
; HistogramPaddingYaml: TotalAccessCount: 4
; HistogramPaddingYaml: AccessHistogramValues: -2 -0 -0 -0 -1 -1
