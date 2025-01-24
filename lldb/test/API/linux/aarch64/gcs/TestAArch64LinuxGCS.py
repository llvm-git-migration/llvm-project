"""
Check that lldb features work when the AArch64 Guarded Control Stack (GCS)
extension is enabled.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxGCSTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_gcs_region(self):
        if not self.isAArch64GCS():
            self.skipTest("Target must support GCS.")

        # This test assumes that we have /proc/<PID>/smaps files
        # that include "VmFlags:" lines.
        # AArch64 kernel config defaults to enabling smaps with
        # PROC_PAGE_MONITOR and "VmFlags" was added in kernel 3.8,
        # before GCS was supported at all.

        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.c",
            line_number("main.c", "// Set break point at this line."),
            num_expected_locations=1,
        )

        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # By now either the program or the system C library enabled GCS and there
        # should be one region marked for use by it (we cannot predict exactly
        # where it will be).
        self.runCmd("memory region --all")
        found_ss = False
        for line in self.res.GetOutput().splitlines():
            if line.strip() == "shadow stack: yes":
                if found_ss:
                    self.fail("Found more than one shadow stack region.")
                found_ss = True

        self.assertTrue(found_ss, "Failed to find a shadow stack region.")

        # Note that we must let the debugee get killed here as it cannot exit
        # cleanly if GCS was manually enabled.

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_gcs_fault(self):
        if not self.isAArch64GCS():
            self.skipTest("Target must support GCS.")

        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect(
            "thread list",
            "Expected stopped by SIGSEGV.",
            substrs=[
                "stopped",
                "stop reason = signal SIGSEGV: control protection fault",
            ],
        )

    def check_gcs_registers(
        self,
        expected_gcs_features_enabled=None,
        expected_gcs_features_locked=None,
        expected_gcspr_el0=None,
    ):
        thread = self.dbg.GetSelectedTarget().process.GetThreadAtIndex(0)
        registerSets = thread.GetFrameAtIndex(0).GetRegisters()
        gcs_registers = registerSets.GetFirstValueByName(
            r"Guarded Control Stack Registers"
        )

        gcs_features_enabled = gcs_registers.GetChildMemberWithName(
            "gcs_features_enabled"
        ).GetValueAsUnsigned()
        if expected_gcs_features_enabled is not None:
            self.assertEqual(expected_gcs_features_enabled, gcs_features_enabled)

        gcs_features_locked = gcs_registers.GetChildMemberWithName(
            "gcs_features_locked"
        ).GetValueAsUnsigned()
        if expected_gcs_features_locked is not None:
            self.assertEqual(expected_gcs_features_locked, gcs_features_locked)

        gcspr_el0 = gcs_registers.GetChildMemberWithName(
            "gcspr_el0"
        ).GetValueAsUnsigned()
        if expected_gcspr_el0 is not None:
            self.assertEqual(expected_gcspr_el0, gcspr_el0)

        return gcs_features_enabled, gcs_features_locked, gcspr_el0

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_gcs_registers(self):
        if not self.isAArch64GCS():
            self.skipTest("Target must support GCS.")

        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        self.runCmd("b test_func")
        self.runCmd("b test_func2")
        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        self.expect("register read --all", substrs=["Guarded Control Stack Registers:"])

        enabled, locked, spr_el0 = self.check_gcs_registers()

        # Features enabled should have at least the enable bit set, it could have
        # others depending on what the C library did.
        self.assertTrue(enabled & 1, "Expected GCS enable bit to be set.")

        # Features locked we cannot predict, we will just assert that it remains
        # the same as we continue.

        # spr_el0 will point to some memory region that is a shadow stack region.
        self.expect(f"memory region {spr_el0}", substrs=["shadow stack: yes"])

        # Continue into test_func2, where the GCS pointer should have been
        # decremented, and the other registers remain the same.
        self.runCmd("continue")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        _, _, spr_el0 = self.check_gcs_registers(enabled, locked, spr_el0 - 8)

        # Modify the control stack pointer to cause a fault.
        spr_el0 += 8
        self.runCmd(f"register write gcspr_el0 {spr_el0}")
        self.expect(
            "register read gcspr_el0", substrs=[f"gcspr_el0 = 0x{spr_el0:016x}"]
        )

        # If we wrote it back correctly, we will now fault but don't pass this
        # signal to the application.
        self.runCmd("process handle SIGSEGV --pass false")
        self.runCmd("continue")

        self.expect(
            "thread list",
            "Expected stopped by SIGSEGV.",
            substrs=[
                "stopped",
                "stop reason = signal SIGSEGV: control protection fault",
            ],
        )

        # Any combination of lock bits could be set. Flip then restore one of them.
        STACK_PUSH = 2
        stack_push = bool((locked >> STACK_PUSH) & 1)
        new_locked = (locked & ~(1 << STACK_PUSH)) | (int(not stack_push) << STACK_PUSH)
        self.runCmd(f"register write gcs_features_locked 0x{new_locked:x}")
        self.expect(
            f"register read gcs_features_locked",
            substrs=[
                f"gcs_features_locked = 0x{new_locked:016x}",
                f"= (PUSH = {(new_locked >> 2) & 1}, WRITE = {(new_locked >> 1) & 1}, ENABLE = {new_locked & 1})",
            ],
        )

        # We could prove the write made it to hardware by trying to prctl to change
        # the feature here, but we cannot know if the libc locked it or not.
        # Given that we know the other registers in the set write correctly, we
        # can assume this one does.

        self.runCmd(f"register write gcs_features_locked 0x{locked:x}")

        # Now to prove we can write gcs_features_enabled, disable GCS and continue
        # past the fault.
        enabled &= ~1
        self.runCmd(f"register write gcs_features_enabled {enabled}")
        self.expect(
            "register read gcs_features_enabled",
            substrs=[
                f"gcs_features_enabled = 0x{enabled:016x}",
                f"= (PUSH = {(enabled >> 2) & 1}, WRITE = {(enabled >> 1) & 1}, ENABLE = {enabled & 1})",
            ],
        )

        self.runCmd("continue")
        self.expect(
            "process status",
            substrs=[
                "exited with status = 0",
            ],
        )

    @skipUnlessPlatform(["linux"])
    def test_gcs_expression_simple(self):
        if not self.isAArch64GCS():
            self.skipTest("Target must support GCS.")

        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        # Break before GCS has been enabled.
        self.runCmd("b main")
        # And after it has been enabled.
        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.c",
            line_number("main.c", "// Set break point at this line."),
            num_expected_locations=1,
        )

        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # GCS has not been enabled yet and the ABI plugin should know not to
        # attempt pushing to the control stack.
        before = self.check_gcs_registers()
        expr_cmd = "p get_gcs_status()"
        self.expect(expr_cmd, substrs=["(unsigned long) 0"])
        self.check_gcs_registers(*before)

        # Continue to when GCS has been enabled.
        self.runCmd("continue")
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # If we fail to setup the GCS entry, we should not leave any of the GCS registers
        # changed. The last thing we do is write a new GCS entry to memory and
        # to simulate the failure of that, temporarily point the GCS to the zero page.
        #
        # We use the value 8 here because LLDB will decrement it by 8 so it points to
        # what we think will be an empty entry on the guarded control stack.
        _, _, original_gcspr = self.check_gcs_registers()
        self.runCmd("register write gcspr_el0 8")
        before = self.check_gcs_registers()
        self.expect(expr_cmd, error=True)
        self.check_gcs_registers(*before)
        # Point to the valid shadow stack region again.
        self.runCmd(f"register write gcspr_el0 {original_gcspr}")

        # This time we do need to push to the GCS and having done so, we can
        # return from this expression without causing a fault.
        before = self.check_gcs_registers()
        self.expect(expr_cmd, substrs=["(unsigned long) 1"])
        self.check_gcs_registers(*before)


    @skipUnlessPlatform(["linux"])
    def test_gcs_expression_disable_gcs(self):
        if not self.isAArch64GCS():
            self.skipTest("Target must support GCS.")

        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        # Break after GCS is enabled.
        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.c",
            line_number("main.c", "// Set break point at this line."),
            num_expected_locations=1,
        )

        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # Unlock all features so the expression can enable them again.
        self.runCmd("register write gcs_features_locked 0")
        # Disable all features, but keep GCS itself enabled.
        PR_SHADOW_STACK_ENABLE = 1
        self.runCmd(f"register write gcs_features_enabled 0x{PR_SHADOW_STACK_ENABLE:x}")

        enabled, locked, spr_el0 = self.check_gcs_registers()
        # We restore everything apart GCS being enabled, as we are not allowed to
        # go from disabled -> enabled via ptrace.
        self.expect("p change_gcs_config(false)", substrs=["true"])
        enabled &= ~1
        self.check_gcs_registers(enabled, locked, spr_el0)

    @skipUnlessPlatform(["linux"])
    def test_gcs_expression_enable_gcs(self):
        if not self.isAArch64GCS():
            self.skipTest("Target must support GCS.")

        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        # Break before GCS is enabled.
        self.runCmd("b main")

        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # Unlock all features so the expression can enable them again.
        self.runCmd("register write gcs_features_locked 0")
        # Disable all features. The program needs PR_SHADOW_STACK_PUSH, but it
        # will enable that itself.
        self.runCmd(f"register write gcs_features_enabled 0")

        enabled, locked, spr_el0 = self.check_gcs_registers()
        self.expect("p change_gcs_config(true)", substrs=["true"])
        # Though we could disable GCS with ptrace, we choose not to to be
        # consistent with the disabled -> enabled behaviour.
        enabled |= 1
        self.check_gcs_registers(enabled, locked, spr_el0)

    @skipIfLLVMTargetMissing("AArch64")
    def test_gcs_core_file(self):
        # To re-generate the core file, build the test file and run it on a
        # machine with GCS enabled. Note that because the kernel decides where
        # the GCS is stored, the value of gcspr_el0 and which memory region it
        # points to may change between runs.

        self.runCmd("target create --core corefile")

        self.expect(
            "bt",
            substrs=["stop reason = SIGSEGV: control protection fault"],
        )

        self.expect(
            "register read --all",
            substrs=[
                "Guarded Control Stack Registers:",
                "gcs_features_enabled = 0x0000000000000001",
                "gcs_features_locked = 0x0000000000000000",
                "gcspr_el0 = 0x0000ffffa83ffff0",
            ],
        )

        # Should get register fields for both. They have the same fields.
        self.expect(
            "register read gcs_features_enabled",
            substrs=["= (PUSH = 0, WRITE = 0, ENABLE = 1)"],
        )
        self.expect(
            "register read gcs_features_locked",
            substrs=["= (PUSH = 0, WRITE = 0, ENABLE = 0)"],
        )

        # Core files do not include /proc/pid/smaps, so we cannot see the
        # shadow stack "ss" flag. gcspr_el0 should at least point to some mapped
        # region.
        self.expect(
            "memory region $gcspr_el0",
            substrs=["[0x0000ffffa8000000-0x0000ffffa8400000) rw-"],
        )
