include(CheckSymbolExists)
include(CMakePushCheckState)

cmake_push_check_state()
set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
check_symbol_exists(gettid "unistd.h" HAVE_GETTID)
if(HAVE_GETTID)
  add_compile_definitions(-DHAVE_GETTID)
endif()
cmake_pop_check_state()

add_lldb_unittest(ProcessElfCoreTests
  ThreadElfCoreTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbUtilityHelpers
    lldbPluginProcessElfCore
    lldbPluginPlatformLinux

    LLVMTestingSupport

  LINK_COMPONENTS
    Support
  )
