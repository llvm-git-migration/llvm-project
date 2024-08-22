import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibCxxStdFunctionRecognizerTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(["libc++"])
    def test_frame_recognizer(self):
        """Test that implementation details details of `std::invoke`"""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        while process.GetState() != lldb.eStateExited:
            self.assertIn("print_num", thread.GetFrameAtIndex(0).GetFunctionName())
            self.process.Continue()
        # # Skip all hidden frames
        # frame_id = 1
        # while frame_id < thread.GetNumFrames() and thread.GetFrameAtIndex(frame_id).IsHidden():
        #     frame_id = frame_id + 1
        # # Expect `std::function<...>::operator()` to be the direct parent of `foo`
        # self.assertIn("::operator()", thread.GetFrameAtIndex(frame_id).GetFunctionName())
        # # And right above that, there should be the `main` frame
        # self.assertIn("main", thread.GetFrameAtIndex(frame_id + 1).GetFunctionName())
