import lldb

SINGLE_INSTANCE_PATTERN = "there_is_only_one_of_me"
DOUBLE_INSTANCE_PATTERN = "there_is_exactly_two_of_me"

def GetAddressRanges(test_base):
    mem_regions = test_base.process.GetMemoryRegions()
    test_base.assertTrue(len(mem_regions) > 0, "Make sure there are memory regions")
    addr_ranges = lldb.SBAddressRangeList()
    for i in range(mem_regions.GetSize()):
        region_info = lldb.SBMemoryRegionInfo()
        if not mem_regions.GetMemoryRegionAtIndex(i, region_info):
            continue
        if not (region_info.IsReadable() and region_info.IsWritable()):
            continue
        if region_info.IsExecutable():
            continue
        if not region_info.GetName() or region_info.GetName() != "[heap]":
            continue

        addr = test_base.target.ResolveLoadAddress(region_info.GetRegionBase())
        addr_range = lldb.SBAddressRange(
            addr,
            region_info.GetRegionEnd() - region_info.GetRegionBase(),
        )
        addr_ranges.Append(addr_range)
    test_base.assertTrue(
        len(addr_ranges) > 0, "Make sure there are valid address ranges"
    )
    return addr_ranges
