set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  AggressiveInstCombine
  Analysis
  AsmParser
  BitWriter
  CFGuard
  CodeGen
  Core
  Coroutines
  Extensions
  IPO
  IRReader
  IRPrinter
  InstCombine
  Instrumentation
  MC
  ObjCARCOpts
  Remarks
  ScalarOpts
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize
  Passes
  )

add_llvm_tool(opt
    PARTIAL_SOURCES_INTENDED
  NewPMDriver.cpp
  opt.cpp

  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )
export_executable_symbols_for_plugins(opt)

if (LLVM_ENABLE_PLUGINS)
add_llvm_tool(opt-printplugin
  NewPMDriver.cpp
  opt.cpp
  PrintPlugin.cpp

  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )
target_link_options(opt-printplugin PUBLIC -Wl,--export-dynamic-symbol=llvmGetPassPluginInfoPrintPlugin)
endif()
