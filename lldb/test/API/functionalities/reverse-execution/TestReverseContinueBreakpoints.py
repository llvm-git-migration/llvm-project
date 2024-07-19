import lldb
import unittest
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbreverse import ReverseTestBase
from lldbsuite.test import lldbutil


class TestReverseContinueBreakpoints(ReverseTestBase):
    @add_test_categories(["dwarf"])
    def test_reverse_continue(self):
        target, process, initial_threads = self.setup_recording()

        # Reverse-continue. We'll stop at the point where we started recording.
        status = process.ReverseContinue()
        self.assertSuccess(status)
        self.expect(
            "thread list",
            STOPPED_DUE_TO_HISTORY_BOUNDARY,
            substrs=["stopped", "stop reason = history boundary"],
        )

        # Continue forward normally until the target exits.
        status = process.Continue()
        self.assertSuccess(status)
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    @add_test_categories(["dwarf"])
    def test_reverse_continue_breakpoint(self):
        target, process, initial_threads = self.setup_recording()

        # Reverse-continue to the function "trigger_breakpoint".
        trigger_bkpt = target.BreakpointCreateByName("trigger_breakpoint", None)
        status = process.ReverseContinue()
        self.assertSuccess(status)
        threads_now = lldbutil.get_threads_stopped_at_breakpoint(process, trigger_bkpt)
        self.assertEqual(threads_now, initial_threads)

    @add_test_categories(["dwarf"])
    def test_reverse_continue_skip_breakpoint(self):
        target, process, initial_threads = self.setup_recording()

        # Reverse-continue, skipping a disabled breakpoint at "trigger_breakpoint".
        trigger_bkpt = target.BreakpointCreateByName("trigger_breakpoint", None)
        trigger_bkpt.SetCondition("0")
        status = process.ReverseContinue()
        self.assertSuccess(status)
        self.expect(
            "thread list",
            STOPPED_DUE_TO_HISTORY_BOUNDARY,
            substrs=["stopped", "stop reason = history boundary"],
        )

    def setup_recording(self):
        """
        Record execution of code between "start_recording" and "stop_recording" breakpoints.

        Returns with the target stopped at "stop_recording", with recording disabled,
        ready to reverse-execute.
        """
        self.build()
        target = self.dbg.CreateTarget("")
        process = self.connect(target)

        # Record execution from the start of the function "start_recording"
        # to the start of the function "stop_recording".
        start_recording_bkpt = target.BreakpointCreateByName("start_recording", None)
        initial_threads = lldbutil.continue_to_breakpoint(process, start_recording_bkpt)
        self.assertEqual(len(initial_threads), 1)
        target.BreakpointDelete(start_recording_bkpt.GetID())
        self.start_recording()
        stop_recording_bkpt = target.BreakpointCreateByName("stop_recording", None)
        lldbutil.continue_to_breakpoint(process, stop_recording_bkpt)
        target.BreakpointDelete(stop_recording_bkpt.GetID())
        self.stop_recording()

        return target, process, initial_threads
