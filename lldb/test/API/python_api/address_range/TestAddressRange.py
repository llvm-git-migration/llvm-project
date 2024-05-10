"""
Test SBAddressRange APIs.
"""

import lldb
from lldbsuite.test.lldbtest import *


class AddressRangeTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_address_range_default(self):
        """Testing default constructor."""
        empty_range = lldb.SBAddressRange()
        self.assertEqual(empty_range.IsValid(), False)

    def test_address_range_construction(self):
        """Make sure the construction and getters work."""
        range = lldb.SBAddressRange(0x00000400, 8)
        self.assertEqual(range.IsValid(), True)
        self.assertEqual(range.GetBaseAddress().GetOffset(), 0x00000400)
        self.assertEqual(range.GetByteSize(), 8)

    def test_address_range_clear(self):
        """Make sure the clear method works."""
        range = lldb.SBAddressRange(0x00000400, 8)
        self.assertEqual(range.IsValid(), True)
        self.assertEqual(range.GetBaseAddress().GetOffset(), 0x00000400)
        self.assertEqual(range.GetByteSize(), 8)

        range.Clear()
        self.assertEqual(range.IsValid(), False)

    def test_function(self):
        """Make sure the range works in SBFunction APIs."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Setup breakpoints in main
        bp = target.BreakpointCreateByName("main", "a.out")
        loc = bp.GetLocationAtIndex(0)
        loc_addr = loc.GetAddress()
        func = loc_addr.GetFunction()
        # block = loc_addr.GetBlock()
        range = func.GetRange()
        # block_ranges = block.GetRangeAtIndex(0)
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
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Setup breakpoints in main
        bp = target.BreakpointCreateByName("main", "a.out")
        loc = bp.GetLocationAtIndex(0)
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

    def test_address_range_list(self):
        """Make sure the SBAddressRangeList works by adding and getting ranges."""
        range1 = lldb.SBAddressRange(0x00000400, 8)
        range2 = lldb.SBAddressRange(0x00000800, 16)
        range3 = lldb.SBAddressRange(0x00001000, 32)

        range_list = lldb.SBAddressRangeList()
        self.assertEqual(range_list.GetSize(), 0)

        range_list.Append(range1)
        range_list.Append(range2)
        range_list.Append(range3)
        self.assertEqual(range_list.GetSize(), 3)

        range1_copy = lldb.SBAddressRange()
        range_list.GetAddressRangeAtIndex(0, range1_copy)
        self.assertEqual(range1.GetByteSize(), range1_copy.GetByteSize())
        self.assertEqual(
            range1.GetBaseAddress().GetOffset(),
            range1_copy.GetBaseAddress().GetOffset(),
        )

        range2_copy = lldb.SBAddressRange()
        range_list.GetAddressRangeAtIndex(1, range2_copy)
        self.assertEqual(range2.GetByteSize(), range2_copy.GetByteSize())
        self.assertEqual(
            range2.GetBaseAddress().GetOffset(),
            range2_copy.GetBaseAddress().GetOffset(),
        )

        range3_copy = lldb.SBAddressRange()
        range_list.GetAddressRangeAtIndex(2, range3_copy)
        self.assertEqual(range3.GetByteSize(), range3_copy.GetByteSize())
        self.assertEqual(
            range3.GetBaseAddress().GetOffset(),
            range3_copy.GetBaseAddress().GetOffset(),
        )

        range_list.Clear()
        self.assertEqual(range_list.GetSize(), 0)
