#include "lldb/Utility/DiagnosticsRendering.h"
#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;
using llvm::StringRef;
namespace {
class ErrorDisplayTest : public ::testing::Test {};

std::string Render(std::vector<DiagnosticDetail> details) {
  StreamString stream;
  RenderDiagnosticDetails(stream, 0, true, details);
  return stream.GetData();
}
} // namespace

TEST_F(ErrorDisplayTest, RenderStatus) {
  {
    DiagnosticDetail::SourceLocation inline_loc;
    inline_loc.in_user_input = true;
    std::string result =
        Render({DiagnosticDetail{inline_loc, eSeverityError, "foo", ""}});
    ASSERT_TRUE(StringRef(result).contains("error:"));
    ASSERT_TRUE(StringRef(result).contains("foo"));
  }

  {
    DiagnosticDetail::SourceLocation loc1 = {FileSpec{"a.c"}, 13,  11, 0,
                                             false,           true};
    DiagnosticDetail::SourceLocation loc2 = {FileSpec{"a.c"}, 13,  13, 0,
                                             false,           true};
    std::string result =
        Render({DiagnosticDetail{loc1, eSeverityError, "1", "1"},
                DiagnosticDetail{loc1, eSeverityError, "2", "3"},
                DiagnosticDetail{loc2, eSeverityError, "3", "3"}});
    ASSERT_TRUE(StringRef(result).contains("error: 1"));
    ASSERT_TRUE(StringRef(result).contains("error: 3"));
    ASSERT_TRUE(StringRef(result).contains("error: 2"));
  }
  {
    DiagnosticDetail::SourceLocation loc1 = {FileSpec{"a.c"}, 1,   20, 0,
                                             false,           true};
    DiagnosticDetail::SourceLocation loc2 = {FileSpec{"a.c"}, 2,   10, 0,
                                             false,           true};
    std::string result =
        Render({DiagnosticDetail{loc2, eSeverityError, "X", "X"},
                DiagnosticDetail{loc1, eSeverityError, "Y", "Y"}});
    ASSERT_LT(StringRef(result).find("Y"), StringRef(result).find("X"));
  }
  {
    DiagnosticDetail::SourceLocation loc1 = {FileSpec{"a.c"}, 1,   1, 3,
                                             false,           true};
    DiagnosticDetail::SourceLocation loc2 = {FileSpec{"a.c"}, 1,   5, 3,
                                             false,           true};
    std::string result =
        Render({DiagnosticDetail{loc1, eSeverityError, "X", "X"},
                DiagnosticDetail{loc2, eSeverityError, "Y", "Y"}});
    auto lines = StringRef(result).split('\n');
    auto line1 = lines.first;
    lines = lines.second.split('\n');
    auto line2 = lines.first;
    lines = lines.second.split('\n');
    auto line3 = lines.first;
    //               1234567
    ASSERT_EQ(line1, "^~~ ^~~");
    ASSERT_EQ(line2, "|   error: Y");
    ASSERT_EQ(line3, "error: X");
  }
}
