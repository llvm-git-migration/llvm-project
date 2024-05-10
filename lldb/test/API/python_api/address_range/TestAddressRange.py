"""
Test SBAddressRange APIs.
"""

import lldb
from lldbsuite.test.lldbtest import *


class AddressRangeTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)

        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.bp1 = target.BreakpointCreateByName("main", "a.out")
        self.bp2 = target.BreakpointCreateByName("foo", "a.out")
        self.bp3 = target.BreakpointCreateByName("bar", "a.out")

        self.assertTrue(self.bp1.IsValid())
        self.assertTrue(self.bp2.IsValid())
        self.assertTrue(self.bp3.IsValid())

        self.addr1 = self.bp1.GetLocationAtIndex(0).GetAddress()
        self.addr2 = self.bp2.GetLocationAtIndex(0).GetAddress()
        self.addr3 = self.bp3.GetLocationAtIndex(0).GetAddress()

        self.assertTrue(self.addr1.IsValid())
        self.assertTrue(self.addr2.IsValid())
        self.assertTrue(self.addr3.IsValid())

    def test_address_range_default(self):
        """Testing default constructor."""
        empty_range = lldb.SBAddressRange()
        self.assertEqual(empty_range.IsValid(), False)

    def test_address_range_construction(self):
        """Make sure the construction and getters work."""
        range = lldb.SBAddressRange(self.addr1, 8)
        self.assertEqual(range.IsValid(), True)
        self.assertEqual(range.GetBaseAddress(), self.addr1)
        self.assertEqual(range.GetByteSize(), 8)

    def test_address_range_clear(self):
        """Make sure the clear method works."""
        range = lldb.SBAddressRange(self.addr1, 8)
        self.assertEqual(range.IsValid(), True)
        self.assertEqual(range.GetBaseAddress(), self.addr1)
        self.assertEqual(range.GetByteSize(), 8)

        range.Clear()
        self.assertEqual(range.IsValid(), False)

    def test_function(self):
        """Make sure the range works in SBFunction APIs."""

        # Setup breakpoints in main
        loc = self.bp1.GetLocationAtIndex(0)
        loc_addr = loc.GetAddress()
        func = loc_addr.GetFunction()
        range = func.GetRange()
        self.assertEqual(
            range.GetByteSize(),
            func.GetEndAddress().GetOffset() - func.GetStartAddress().GetOffset(),
        )
        self.assertEqual(
            range.GetBaseAddress().GetOffset(),
            func.GetStartAddress().GetOffset(),
        )

    def test_block(self):
        """Make sure the range works in SBBlock APIs."""
        loc = self.bp1.GetLocationAtIndex(0)
        loc_addr = loc.GetAddress()
        block = loc_addr.GetBlock()
        range = block.GetRangeAtIndex(0)
        self.assertEqual(
            range.GetByteSize(),
            block.GetRangeEndAddress(0).GetOffset()
            - block.GetRangeStartAddress(0).GetOffset(),
        )
        self.assertEqual(
            range.GetBaseAddress().GetOffset(),
            block.GetRangeStartAddress(0).GetOffset(),
        )

        ranges = block.GetRanges()
        self.assertEqual(ranges.GetSize(), 1)
        self.assertEqual(ranges.GetAddressRangeAtIndex(0), range)

    def test_address_range_list(self):
        """Make sure the SBAddressRangeList works by adding and getting ranges."""
        range1 = lldb.SBAddressRange(self.addr1, 8)
        range2 = lldb.SBAddressRange(self.addr2, 16)
        range3 = lldb.SBAddressRange(self.addr3, 32)

        range_list = lldb.SBAddressRangeList()
        self.assertEqual(range_list.GetSize(), 0)

        range_list.Append(range1)
        range_list.Append(range2)
        range_list.Append(range3)
        self.assertEqual(range_list.GetSize(), 3)

        range1_copy = range_list.GetAddressRangeAtIndex(0)
        self.assertEqual(range1.GetByteSize(), range1_copy.GetByteSize())
        self.assertEqual(
            range1.GetBaseAddress().GetOffset(),
            range1_copy.GetBaseAddress().GetOffset(),
        )

        range2_copy = range_list.GetAddressRangeAtIndex(1)
        self.assertEqual(range2.GetByteSize(), range2_copy.GetByteSize())
        self.assertEqual(
            range2.GetBaseAddress().GetOffset(),
            range2_copy.GetBaseAddress().GetOffset(),
        )

        range3_copy = range_list.GetAddressRangeAtIndex(2)
        self.assertEqual(range3.GetByteSize(), range3_copy.GetByteSize())
        self.assertEqual(
            range3.GetBaseAddress().GetOffset(),
            range3_copy.GetBaseAddress().GetOffset(),
        )

        range_list.Clear()
        self.assertEqual(range_list.GetSize(), 0)

    def test_address_range_list_len(self):
        """Make sure the len() operator works."""
        range = lldb.SBAddressRange(self.addr1, 8)

        range_list = lldb.SBAddressRangeList()
        self.assertEqual(len(range_list), 0)

        range_list.Append(range)
        self.assertEqual(len(range_list), 1)

    def test_address_range_list_iterator(self):
        """Make sure the SBAddressRangeList iterator works."""
        range1 = lldb.SBAddressRange(self.addr1, 8)
        range2 = lldb.SBAddressRange(self.addr2, 16)
        range3 = lldb.SBAddressRange(self.addr3, 32)

        range_list = lldb.SBAddressRangeList()
        range_list.Append(range1)
        range_list.Append(range2)
        range_list.Append(range3)
        self.assertEqual(range_list.GetSize(), 3)

        # Test the iterator
        for range in range_list:
            self.assertTrue(range.IsValid())

    def test_address_range_print(self):
        """Make sure the SBAddressRange can be printed."""
        range = lldb.SBAddressRange(self.addr1, 8)

        range_str = str(range)

        offset_str = hex(self.addr1.GetOffset())[2:]
        self.assertIn(offset_str, range_str)

        byte_size_str = hex(range.GetByteSize())[2:]
        self.assertIn(byte_size_str, range_str)

    def test_address_range_list_print(self):
        """Make sure the SBAddressRangeList can be printed."""
        range1 = lldb.SBAddressRange(self.addr1, 8)
        range2 = lldb.SBAddressRange(self.addr2, 16)
        range3 = lldb.SBAddressRange(self.addr3, 32)

        range_list = lldb.SBAddressRangeList()
        self.assertEqual(range_list.GetSize(), 0)

        range_list.Append(range1)
        range_list.Append(range2)
        range_list.Append(range3)
        self.assertEqual(range_list.GetSize(), 3)

        range_list_str = str(range_list)
        self.assertIn("3 address ranges:", range_list_str)
        self.assertEqual(range_list_str.count("AddressRange"), 3)
