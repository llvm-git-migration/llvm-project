--- |
  ; ModuleID = '/home/shiltian/Documents/vscode/llvm-project/llvm/test/CodeGen/AMDGPU/GlobalISel/madmix-constant-bus-violation.mir'
  source_filename = "/home/shiltian/Documents/vscode/llvm-project/llvm/test/CodeGen/AMDGPU/GlobalISel/madmix-constant-bus-violation.mir"
  target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
  target triple = "amdgcn"
  
  define void @foo() #0 {
  entry:
    unreachable
  }
  
  attributes #0 = { "target-cpu"="gfx900" }

...
---
name:            foo
alignment:       1
exposesReturnsTwice: false
legalized:       true
regBankSelected: true
selected:        true
failedISel:      false
tracksRegLiveness: false
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         false
callsEHReturn:   false
callsUnwindInit: false
hasEHCatchret:   false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: sreg_32, preferred-register: '' }
  - { id: 1, class: sreg_32, preferred-register: '' }
  - { id: 2, class: sreg_32, preferred-register: '' }
  - { id: 3, class: sreg_32, preferred-register: '' }
  - { id: 4, class: sreg_32, preferred-register: '' }
  - { id: 5, class: sreg_32, preferred-register: '' }
  - { id: 6, class: sreg_32, preferred-register: '' }
  - { id: 7, class: vgpr_32, preferred-register: '' }
  - { id: 8, class: vgpr, preferred-register: '' }
  - { id: 9, class: vgpr_32, preferred-register: '' }
  - { id: 10, class: vgpr, preferred-register: '' }
  - { id: 11, class: vgpr, preferred-register: '' }
  - { id: 12, class: vgpr_32, preferred-register: '' }
  - { id: 13, class: sreg_32, preferred-register: '' }
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
  savePoint:       ''
  restorePoint:    ''
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: false
    fp32-output-denormals: false
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       8
  vgprForAGPRCopy: ''
  sgprForEXECCopy: '$sgpr100_sgpr101'
  longBranchReservedReg: ''
  hasInitWholeWave: false
body:             |
  bb.0:
    %0:sreg_32 = COPY $sgpr0
    %1:sreg_32 = COPY $sgpr1
    %2:sreg_32 = S_MOV_B32 16
    %3:sreg_32 = S_LSHR_B32 %0, %2, implicit-def dead $scc
    %5:sreg_32 = S_LSHR_B32 %1, %2, implicit-def dead $scc
    %7:vgpr_32 = COPY %3
    %9:vgpr_32 = COPY %5
    %12:vgpr_32 = V_MAD_MIX_F32 9, %9, 8, %9, 8, %7, 0, 0, 0, implicit $mode, implicit $exec
    %13:sreg_32 = V_READFIRSTLANE_B32 %12, implicit $exec
    $sgpr0 = COPY %13
    SI_RETURN_TO_EPILOG implicit $sgpr0

...
