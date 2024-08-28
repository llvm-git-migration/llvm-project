#include "Commands/CommandObjectExpression.h"

#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;
using llvm::StringRef;
namespace {
class ErrorDisplayTest : public ::testing::Test {};
} // namespace

static std::string Render(llvm::SmallVector<Status::Detail> details) {
  Status error = Status::FromExpressionErrorDetails(eExpressionCompleted,
                                                    std::move(details));
  return CommandObjectExpression::RenderError(0, error);
}

TEST_F(ErrorDisplayTest, RenderStatus) {
  Status::Detail::SourceLocation inline_loc;
  inline_loc.in_user_input = true;
  {
    std::string result =
      Render({Status::Detail{inline_loc, eSeverityError, "foo", ""}});
    ASSERT_TRUE(StringRef(result).contains("error:"));
    ASSERT_TRUE(StringRef(result).contains("foo"));
  }
}
