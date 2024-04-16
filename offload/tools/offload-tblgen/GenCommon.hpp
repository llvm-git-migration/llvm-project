#pragma once

#include "RecordTypes.hpp"
#include "llvm/Support/FormatVariadic.h"

constexpr auto CommentsHeader = R"(
///////////////////////////////////////////////////////////////////////////////
)";

constexpr auto CommentsBreak = "///\n";

constexpr auto PrefixLower = "ol";
constexpr auto PrefixUpper = "OL";

static std::string
MakeParamComment(const llvm::offload::tblgen::ParamRec &Param) {
  return llvm::formatv("///< {0}{1}{2} {3}", (Param.isIn() ? "[in]" : ""),
                       (Param.isOut() ? "[out]" : ""),
                       (Param.isOpt() ? "[optional]" : ""), Param.getDesc());
}
