add_lldb_unittest(LLDBBreakpointTests
  BreakpointIDTest.cpp
  WatchpointAlgorithmsTests.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    LLVMTestingSupport
    lldbUtilityHelpers
    lldbPluginPlatformMacOSX
  LINK_COMPONENTS
    Support
  )
