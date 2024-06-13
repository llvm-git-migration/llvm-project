//===-- TestBreakpointSetCallback.cpp
//--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Progress.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-types.h"
#include "gtest/gtest.h"
#include <iostream>
#include <memory>
#include <mutex>

using namespace lldb_private;
using namespace lldb;

#define EXPECTED_BREAKPOINT_ID 1
#define EXPECTED_BREAKPOINT_LOCATION_ID 0

class BreakpointSetCallbackTest : public ::testing::Test {
public:
  static void CheckCallbackArgs(void *baton, StoppointCallbackContext *context,
                                lldb::user_id_t break_id,
                                lldb::user_id_t break_loc_id,
                                lldb::user_id_t expected_breakpoint_id,
                                lldb::user_id_t expected_breakpoint_loc_id,
                                TargetSP expected_target_sp) {
    EXPECT_EQ(context->exe_ctx_ref.GetTargetSP(), expected_target_sp);
    EXPECT_EQ(baton, "hello");
    EXPECT_EQ(break_id, expected_breakpoint_id);
    EXPECT_EQ(break_loc_id, expected_breakpoint_loc_id);
  }

protected:
  void SetUp() override {
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
  };

  DebuggerSP m_debugger_sp;
  PlatformSP m_platform_sp;
  BreakpointSP m_breakpoint_sp;
  SubsystemRAII<FileSystem, HostInfo, PlatformMacOSX, ProgressManager>
      subsystems;
};

TEST_F(BreakpointSetCallbackTest, TestBreakpointSetCallback) {
  void *baton = (void *)"hello";
  // Set up the debugger, make sure that was done properly.
  TargetSP m_target_sp;
  ArchSpec arch("x86_64-apple-macosx-");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  m_debugger_sp = Debugger::CreateInstance();

  // Create target
  m_debugger_sp->GetTargetList().CreateTarget(*m_debugger_sp, "", arch,
                                              lldb_private::eLoadDependentsNo,
                                              m_platform_sp, m_target_sp);

  // Create breakpoint
  m_breakpoint_sp = m_target_sp->CreateBreakpoint(0xDEADBEEF, false, false);

  m_breakpoint_sp->SetCallback(
      [m_target_sp](void *baton, StoppointCallbackContext *context,
                    lldb::user_id_t break_id, lldb::user_id_t break_loc_id) {
        CheckCallbackArgs(baton, context, break_id, break_loc_id,
                          EXPECTED_BREAKPOINT_ID,
                          EXPECTED_BREAKPOINT_LOCATION_ID, m_target_sp);
        return true;
      },
      baton, true);
  ExecutionContext m_exe_ctx(m_target_sp, false);
  StoppointCallbackContext context(nullptr, m_exe_ctx, true);
  m_breakpoint_sp->InvokeCallback(&context, 0);
}
