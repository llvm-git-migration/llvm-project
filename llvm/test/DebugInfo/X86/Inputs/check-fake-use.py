#!/usr/bin/python3

# Parsing dwarfdump's output to determine whether the location list for the
# parameter "b" covers all of the function. The script is written in form of a
# state machine and expects that dwarfdump output adheres to a certain order:
# 1) The .debug_info section must appear before the .debug_loc section.
# 2) The DW_AT_location attribute must appear before the parameter's name in the
#    formal parameter DIE.
#
import re
import sys

from enum import IntEnum, auto

DebugInfoPattern = r"\.debug_info contents:"
SubprogramPattern = r"^0x[0-9a-f]+:\s+DW_TAG_subprogram"
ProloguePattern = r"^\s*0x([0-9a-f]+)\s.+prologue_end"
EpiloguePattern = r"^\s*0x([0-9a-f]+)\s.+epilogue_begin"
FormalPattern = r"^0x[0-9a-f]+:\s+DW_TAG_formal_parameter"
LocationPattern = r"DW_AT_location\s+\[DW_FORM_sec_offset\].*0x([a-f0-9]+)"
DebugLocPattern = r'\[0x([a-f0-9]+),\s+0x([a-f0-9]+)\) ".text":'

# States
class States(IntEnum):
    LookingForDebugInfo = 0
    LookingForSubProgram = auto()
    LookingForFormal = auto()
    LookingForLocation = auto()
    DebugLocations = auto()
    LookingForPrologue = auto()
    LookingForEpilogue = auto()
    AllDone = auto()

# For each state, the state table contains 3-item sublists with the following
# entries:
# 1) The regex pattern we use in each state.
# 2) The state we enter when we have a successful match for the current pattern.
# 3) The state we enter when we do not have a successful match for the
#    current pattern.
StateTable = [
    # LookingForDebugInfo
    [DebugInfoPattern, States.LookingForSubProgram, States.LookingForDebugInfo],
    # LookingForSubProgram
    [SubprogramPattern, States.LookingForFormal, States.LookingForSubProgram],
    # LookingForFormal
    [FormalPattern, States.LookingForLocation, States.LookingForFormal],
    # LookingForLocation
    [LocationPattern, States.DebugLocations, States.LookingForFormal],
    # DebugLocations
    [DebugLocPattern, States.DebugLocations, States.LookingForPrologue],
    # LookingForPrologue
    [ProloguePattern, States.LookingForEpilogue, States.LookingForPrologue],
    # LookingForEpilogue
    [EpiloguePattern, States.AllDone, States.LookingForEpilogue],
    # AllDone
    [None, States.AllDone, States.AllDone],
]

# Symbolic indices
StatePattern = 0
NextState = 1
FailState = 2

State = States.LookingForDebugInfo
FirstBeginOffset = -1
ContinuousLocation = True
LocationBreak = ()
LocationRanges = []

# Read output from file provided as command arg
with open(sys.argv[1], "r") as dwarf_dump_file:
    for line in dwarf_dump_file:
        if State == States.AllDone:
            break
        Pattern = StateTable[State][StatePattern]
        m = re.search(Pattern, line)
        if m:
            # Match. Depending on the state, we extract various values.
            if State == States.LookingForPrologue:
                PrologueEnd = int(m.group(1), 16)
            elif State == States.LookingForEpilogue:
                EpilogueBegin = int(m.group(1), 16)
            elif State == States.DebugLocations:
                # Extract the range values
                if FirstBeginOffset == -1:
                    FirstBeginOffset = int(m.group(1), 16)
                else:
                    NewBeginOffset = int(m.group(1), 16)
                    if NewBeginOffset != EndOffset:
                        ContinuousLocation = False
                        LocationBreak = (EndOffset, NewBeginOffset)
                EndOffset = int(m.group(2), 16)
            State = StateTable[State][NextState]
        else:
            State = StateTable[State][FailState]

Success = True

# Check that the first entry start with 0 and that the last ending address
# in our location list is close to the high pc of the subprogram.
if State != States.AllDone:
    print("Error in expected sequence of DWARF information:")
    print(" State = %d\n" % State)
    Success = False
elif FirstBeginOffset == -1:
    print("Location list for 'b' not found, did the debug info format change?")
    Success = False
elif not ContinuousLocation:
    print("Location list for 'b' is discontinuous from [0x%x, 0x%x)" % LocationBreak)
    Success = False
elif FirstBeginOffset > PrologueEnd or EndOffset < EpilogueBegin:
    print("Location list for 'b' does not cover the whole function:")
    print(
        "Prologue to Epilogue = [0x%x, 0x%x), Location range = [0x%x, 0x%x)"
        % (PrologueEnd, EpilogueBegin, FirstBeginOffset, EndOffset)
    )
    Success = False

sys.exit(not Success)
