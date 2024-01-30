//===-- Implementation of PublicAPICommand --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PublicAPICommand.h"

#include "utils/LibcTableGenUtil/APIIndexer.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <llvm/ADT/STLExtras.h>

// Text blocks for macro definitions and type decls can be indented to
// suit the surrounding tablegen listing. We need to dedent such blocks
// before writing them out.
static void dedentAndWrite(llvm::StringRef Text, llvm::raw_ostream &OS) {
  llvm::SmallVector<llvm::StringRef, 10> Lines;
  llvm::SplitString(Text, Lines, "\n");
  size_t shortest_indent = 1024;
  for (llvm::StringRef L : Lines) {
    llvm::StringRef Indent = L.take_while([](char c) { return c == ' '; });
    size_t IndentSize = Indent.size();
    if (Indent.size() == L.size()) {
      // Line is all spaces so no point noting the indent.
      continue;
    }
    if (IndentSize < shortest_indent)
      shortest_indent = IndentSize;
  }
  for (llvm::StringRef L : Lines) {
    if (L.size() >= shortest_indent)
      OS << L.drop_front(shortest_indent) << '\n';
  }
}

static std::string getTypeHdrName(const std::string &Name) {
  llvm::SmallVector<llvm::StringRef> Parts;
  llvm::SplitString(llvm::StringRef(Name), Parts);
  return llvm::join(Parts.begin(), Parts.end(), "_");
}

namespace llvm_libc {

void writeAPIFromIndex(APIIndexer &G,
                       std::vector<std::string> EntrypointNameList,
                       llvm::raw_ostream &OS) {
  for (auto &Pair : G.MacroDefsMap) {
    const std::string &Name = Pair.first;
    if (G.MacroSpecMap.find(Name) == G.MacroSpecMap.end())
      llvm::PrintFatalError(Name + " not found in any standard spec.\n");

    llvm::Record *MacroDef = Pair.second;
    dedentAndWrite(MacroDef->getValueAsString("Defn"), OS);

    OS << '\n';
  }

  for (auto &TypeName : G.RequiredTypes) {
    if (G.TypeSpecMap.find(TypeName) == G.TypeSpecMap.end())
      llvm::PrintFatalError(TypeName + " not found in any standard spec.\n");
    OS << "#include <llvm-libc-types/" << getTypeHdrName(TypeName) << ".h>\n";
  }
  OS << '\n';

  if (G.Enumerations.size() != 0)
    OS << "enum {" << '\n';
  for (const auto &Name : G.Enumerations) {
    if (G.EnumerationSpecMap.find(Name) == G.EnumerationSpecMap.end())
      llvm::PrintFatalError(
          Name + " is not listed as an enumeration in any standard spec.\n");

    llvm::Record *EnumerationSpec = G.EnumerationSpecMap[Name];
    OS << "  " << EnumerationSpec->getValueAsString("Name");
    auto Value = EnumerationSpec->getValueAsString("Value");
    if (Value == "__default__") {
      OS << ",\n";
    } else {
      OS << " = " << Value << ",\n";
    }
  }
  if (G.Enumerations.size() != 0)
    OS << "};\n\n";

  // declare macros for attributes
  llvm::DenseMap<llvm::StringRef, llvm::Record *> MacroAttr;
  for (auto &Name : EntrypointNameList) {
    if (G.FunctionSpecMap.find(Name) == G.FunctionSpecMap.end()) {
      continue;
    }
    llvm::Record *FunctionSpec = G.FunctionSpecMap[Name];
    auto Attributes = FunctionSpec->getValueAsListOfDefs("Attributes");
    for (auto *Attr : Attributes) {
      MacroAttr[Attr->getValueAsString("Macro")] = Attr;
    }
  }

  auto GetStyle = [](llvm::Record *Instance) {
    auto Style = Instance->getValueAsString("Style");
    if (Style == "cxx11")
      return AttributeStyle::Cxx11;
    if (Style == "gnu")
      return AttributeStyle::Gnu;
    return AttributeStyle::Declspec;
  };

  auto GetNamespace = [](llvm::Record *Instance) {
    auto Namespace = Instance->getValueAsString("Namespace");
    // Empty namespace is likely to be most standard-compliant.
    if (Namespace.empty())
      return AttributeNamespace::None;
    // Dispatch clang version before gnu version.
    if (Namespace == "clang")
      return AttributeNamespace::Clang;
    return AttributeNamespace::Gnu;
  };

  for (auto &[Macro, Attr] : MacroAttr) {
    auto Instances = Attr->getValueAsListOfDefs("Instances");
    llvm::SmallVector<std::pair<AttributeStyle, llvm::Record *>> Styles;
    std::transform(Instances.begin(), Instances.end(),
                   std::back_inserter(Styles),
                   [&](llvm::Record *Instance)
                       -> std::pair<AttributeStyle, llvm::Record *> {
                     auto Style = GetStyle(Instance);
                     return {Style, Instance};
                   });
    // Effectively sort on the first field
    std::sort(Styles.begin(), Styles.end(), [&](auto &a, auto &b) {
      if (a.first == AttributeStyle::Cxx11 && b.first == AttributeStyle::Cxx11)
        return GetNamespace(a.second) < GetNamespace(b.second);
      return a.first < b.first;
    });
    for (auto &[Style, Instance] : Styles) {
      if (Style == AttributeStyle::Cxx11) {
        OS << "#if !defined(" << Macro << ") && defined(__cplusplus)";
        auto Namespace = GetNamespace(Instance);
        if (Namespace == AttributeNamespace::Clang)
          OS << " && defined(__clang__)\n";
        else if (Namespace == AttributeNamespace::Gnu)
          OS << " && defined(__GNUC__)\n";
        else
          OS << '\n';
        OS << "#define " << Macro << " [[";
        if (Namespace == AttributeNamespace::Clang)
          OS << "clang::";
        else if (Namespace == AttributeNamespace::Gnu)
          OS << "gnu::";
        OS << Instance->getValueAsString("Attr") << "]]\n";
        OS << "#endif\n";
      }
      if (Style == AttributeStyle::Gnu) {
        OS << "#if !defined(" << Macro << ") && defined(__GNUC__)\n";
        OS << "#define " << Macro << " __attribute__((";
        OS << Instance->getValueAsString("Attr") << "))\n";
        OS << "#endif\n";
      }
      if (Style == AttributeStyle::Declspec) {
        OS << "#if !defined(" << Macro << ") && defined(_MSC_VER)\n";
        OS << "#define " << Macro << " __declspec(";
        OS << Instance->getValueAsString("Attr") << ")\n";
        OS << "#endif\n";
      }
    }
    OS << "#if !defined(" << Macro << ")\n";
    OS << "#define " << Macro << '\n';
    OS << "#endif\n";
  }

  if (!MacroAttr.empty())
    OS << '\n';

  OS << "__BEGIN_C_DECLS\n\n";
  for (auto &Name : EntrypointNameList) {
    if (G.FunctionSpecMap.find(Name) == G.FunctionSpecMap.end()) {
      continue; // Functions that aren't in this header file are skipped as
                // opposed to erroring out because the list of functions being
                // iterated over is the complete list of functions with
                // entrypoints. Thus this is filtering out the functions that
                // don't go to this header file, whereas the other, similar
                // conditionals above are more of a sanity check.
    }

    llvm::Record *FunctionSpec = G.FunctionSpecMap[Name];
    llvm::Record *RetValSpec = FunctionSpec->getValueAsDef("Return");
    llvm::Record *ReturnType = RetValSpec->getValueAsDef("ReturnType");

    auto Attributes = FunctionSpec->getValueAsListOfDefs("Attributes");
    llvm::interleave(
        Attributes.begin(), Attributes.end(),
        [&](llvm::Record *Attr) { OS << Attr->getValueAsString("Macro"); },
        [&]() { OS << ' '; });
    if (!Attributes.empty())
      OS << ' ';

    OS << G.getTypeAsString(ReturnType) << " " << Name << "(";

    auto ArgsList = FunctionSpec->getValueAsListOfDefs("Args");
    for (size_t i = 0; i < ArgsList.size(); ++i) {
      llvm::Record *ArgType = ArgsList[i]->getValueAsDef("ArgType");
      OS << G.getTypeAsString(ArgType);
      if (i < ArgsList.size() - 1)
        OS << ", ";
    }

    OS << ") __NOEXCEPT;\n\n";
  }

  // Make another pass over entrypoints to emit object declarations.
  for (const auto &Name : EntrypointNameList) {
    if (G.ObjectSpecMap.find(Name) == G.ObjectSpecMap.end())
      continue;
    llvm::Record *ObjectSpec = G.ObjectSpecMap[Name];
    auto Type = ObjectSpec->getValueAsString("Type");
    OS << "extern " << Type << " " << Name << ";\n";
  }
  OS << "__END_C_DECLS\n";

  // undef the macros
  for (auto &[Macro, Attr] : MacroAttr)
    OS << "\n#undef " << Macro << '\n';
}

void writePublicAPI(llvm::raw_ostream &OS, llvm::RecordKeeper &Records) {}

const char PublicAPICommand::Name[] = "public_api";

void PublicAPICommand::run(llvm::raw_ostream &OS, const ArgVector &Args,
                           llvm::StringRef StdHeader,
                           llvm::RecordKeeper &Records,
                           const Command::ErrorReporter &Reporter) const {
  if (Args.size() != 0) {
    Reporter.printFatalError("public_api command does not take any arguments.");
  }

  APIIndexer G(StdHeader, Records);
  writeAPIFromIndex(G, EntrypointNameList, OS);
}

} // namespace llvm_libc
