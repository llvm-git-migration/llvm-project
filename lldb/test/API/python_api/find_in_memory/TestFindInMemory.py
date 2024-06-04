"""
Test Process::FindInMemory.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from address_ranges_helper import *


class FindInMemoryTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)

        self.build()
        (
            self.target,
            self.process,
            self.thread,
            self.bp,
        ) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        self.assertTrue(self.bp.IsValid())

    def test_find_in_memory_ok(self):
        """Make sure a match exists in the heap memory and the right address ranges are provided"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        error = lldb.SBError()
        addr = self.process.FindInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            GetStackRange(self),
            1,
            error,
        )

        self.assertSuccess(error)
        self.assertNotEqual(addr, lldb.LLDB_INVALID_ADDRESS)

    def test_find_in_memory_double_instance_ok(self):
        """Make sure a match exists in the heap memory and the right address ranges are provided"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        error = lldb.SBError()
        addr = self.process.FindInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            GetHeapRanges(self)[0],
            1,
            error,
        )

        self.assertSuccess(error)
        self.assertNotEqual(addr, lldb.LLDB_INVALID_ADDRESS)

    def test_find_in_memory_invalid_alignment(self):
        """Make sure the alignment 0 is failing"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        error = lldb.SBError()
        addr = self.process.FindInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            GetStackRange(self),
            0,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(addr, lldb.LLDB_INVALID_ADDRESS)

    def test_find_in_memory_invalid_address_range(self):
        """Make sure invalid address range is failing"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        error = lldb.SBError()
        addr = self.process.FindInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            lldb.SBAddressRange(),
            1,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(addr, lldb.LLDB_INVALID_ADDRESS)

    def test_find_in_memory_invalid_buffer(self):
        """Make sure the empty buffer is failing"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        error = lldb.SBError()
        addr = self.process.FindInMemory(
            "",
            GetStackRange(self),
            1,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(addr, lldb.LLDB_INVALID_ADDRESS)
