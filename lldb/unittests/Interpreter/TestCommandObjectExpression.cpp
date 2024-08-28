#include "lldb/Expression/DiagnosticManager.h"
#include "gtest/gtest.h"

namespace lldb_private {
std::string RenderDiagnosticDetails(std::optional<uint16_t> offset_in_command,
                                    llvm::ArrayRef<DiagnosticDetail> details);
}

using namespace lldb_private;
using namespace lldb;
using llvm::StringRef;
namespace {
class ErrorDisplayTest : public ::testing::Test {};
} // namespace

static std::string Render(std::vector<DiagnosticDetail> details) {
  return RenderDiagnosticDetails(0, details);
}

TEST_F(ErrorDisplayTest, RenderStatus) {
  DiagnosticDetail::SourceLocation inline_loc;
  inline_loc.in_user_input = true;
  {
    std::string result =
        Render({DiagnosticDetail{inline_loc, eSeverityError, "foo", ""}});
    ASSERT_TRUE(StringRef(result).contains("error:"));
    ASSERT_TRUE(StringRef(result).contains("foo"));
  }
}
