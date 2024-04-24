//===- LLDBPropertyDefEmitter.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These tablegen backends emits LLDB's PropertyDefinition values.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <llvm/ADT/StringRef.h>
namespace lldb_private {
int EmitSBAPIDWARFEnum(int argc, char **argv) {
  std::string InputFilename;
  std::string OutputFilename;
  // This command line option parser is as robust as the worst shell script.
  for (int i = 0; i < argc; ++i) {
    if (llvm::StringRef(argv[i]).ends_with("Dwarf.def"))
      InputFilename = std::string(argv[i]);
    if (llvm::StringRef(argv[i]) == "-o" && i + 1 < argc)
      OutputFilename = std::string(argv[i+1]);
  }
  std::ifstream input(InputFilename);
  std::ofstream output(OutputFilename);
  output << "// Do not include this file directly.\n";
  output << "#ifndef HANDLE_DW_LNAME\n";
  output << "#error \"Missing macro definition\"\n";
  output << "#endif\n";
  std::string line;
  while (std::getline(input, line)) {
    if (llvm::StringRef(line).starts_with("HANDLE_DW_LNAME"))
      output << line << '\n';
  }
  output << "#undef HANDLE_DW_LNAME\n";
  return 0;
}
} // namespace lldb_private
