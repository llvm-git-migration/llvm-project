"""
Test Process::FindInMemory.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from address_ranges_helper import GetAddressRanges
from address_ranges_helper import SINGLE_INSTANCE_PATTERN


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

        addr_ranges = GetAddressRanges(self)
        for addr_range in addr_ranges:
            error = lldb.SBError()
            addr = self.process.FindInMemory(
                SINGLE_INSTANCE_PATTERN,
                addr_range,
                1,
                error,
            )

            if addr != lldb.LLDB_INVALID_ADDRESS:
                break

        self.assertSuccess(error)
        self.assertNotEqual(addr, lldb.LLDB_INVALID_ADDRESS)

    def test_find_in_memory_invalid_alignment(self):
        """Make sure the alignment 0 is failing"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetAddressRanges(self)

        error = lldb.SBError()
        addr = self.process.FindInMemory(
            SINGLE_INSTANCE_PATTERN,
            addr_ranges[0],
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
            SINGLE_INSTANCE_PATTERN,
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

        addr_ranges = GetAddressRanges(self)

        error = lldb.SBError()
        addr = self.process.FindInMemory(
            "",
            addr_ranges[0],
            1,
            error,
        )

        error = lldb.SBError()
        self.assertEqual(addr, lldb.LLDB_INVALID_ADDRESS)
