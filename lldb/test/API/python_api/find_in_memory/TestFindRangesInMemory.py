"""
Test Process::FindRangesInMemory.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from address_ranges_helper import GetAddressRanges
from address_ranges_helper import SINGLE_INSTANCE_PATTERN
from address_ranges_helper import DOUBLE_INSTANCE_PATTERN


class FindRangesInMemoryTestCase(TestBase):
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

    def test_find_ranges_in_memory_two_matches(self):
        """Make sure two matches exist in the heap memory and the right address ranges are provided"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetAddressRanges(self)

        matches = self.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN,
            addr_ranges,
            1,
            10,
        )

        self.assertEqual(matches.GetSize(), 2)

    def test_find_ranges_in_memory_one_match(self):
        """Make sure exactly one match exists in the heap memory and the right address ranges are provided"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetAddressRanges(self)

        matches = self.process.FindRangesInMemory(
            SINGLE_INSTANCE_PATTERN,
            addr_ranges,
            1,
            10,
        )

        self.assertEqual(matches.GetSize(), 1)

    def test_find_ranges_in_memory_one_match_max(self):
        """Make sure at least one matche exists in the heap memory and the right address ranges are provided"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetAddressRanges(self)

        matches = self.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN,
            addr_ranges,
            1,
            1,
        )

        self.assertEqual(matches.GetSize(), 1)

    def test_find_ranges_in_memory_invalid_alignment(self):
        """Make sure the alignment 0 is failing"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetAddressRanges(self)

        matches = self.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN,
            addr_ranges,
            0,
            10,
        )

        self.assertEqual(matches.GetSize(), 0)

    def test_find_ranges_in_memory_empty_ranges(self):
        """Make sure the empty ranges is failing"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = lldb.SBAddressRangeList()
        matches = self.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN,
            addr_ranges,
            1,
            10,
        )

        self.assertEqual(matches.GetSize(), 0)

    def test_find_ranges_in_memory_invalid_buffer(self):
        """Make sure the empty buffer is failing"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetAddressRanges(self)

        matches = self.process.FindRangesInMemory(
            "",
            addr_ranges,
            1,
            10,
        )

        self.assertEqual(matches.GetSize(), 0)
