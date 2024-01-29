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
                       llvm::raw_ostream &OS, AttributeStyle PreferedStyle) {
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

    auto GetStyle = [](llvm::Record *Instance) {
      auto Style = Instance->getValueAsString("Style");
      if (Style == "gnu")
        return AttributeStyle::Gnu;
      if (Style == "c23")
        return AttributeStyle::C23;
      if (Style == "declspec")
        return AttributeStyle::Declspec;
      return AttributeStyle::None;
    };

    if (PreferedStyle != AttributeStyle::None) {
      auto Attributes = FunctionSpec->getValueAsListOfDefs("Attributes");
      llvm::SmallVector<llvm::Record *> Attrs;
      for (auto *Attr : Attributes) {
        auto Instances = Attr->getValueAsListOfDefs("Instances");
        for (auto *Instance : Instances) {
          if (GetStyle(Instance) == PreferedStyle) {
            Attrs.push_back(Instance);
          }
        }
      }

      if (Attrs.size() != 0) {
        if (PreferedStyle == AttributeStyle::Gnu) {
          OS << "__attribute__((";
          llvm::interleaveComma(Attrs, OS, [&](llvm::Record *Instance) {
            OS << Instance->getValueAsString("Attr");
          });
          OS << ")) ";
        }

        if (PreferedStyle == AttributeStyle::C23) {
          OS << "__attribute__((";
          llvm::interleaveComma(Attrs, OS, [&](llvm::Record *Instance) {
            auto Namespace = Instance->getValueAsString("Namespace");
            if (Namespace != "")
              OS << Namespace << "::";
            OS << Instance->getValueAsString("Attr");
          });
          OS << ")) ";
        }

        if (PreferedStyle == AttributeStyle::Declspec) {
          OS << "__declspec(";
          llvm::interleave(
              Attrs.begin(), Attrs.end(),
              [&](llvm::Record *Instance) {
                OS << Instance->getValueAsString("Attr");
              },
              [&]() { OS << ' '; });
          OS << ") ";
        }
      }
    }

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
}

void writePublicAPI(llvm::raw_ostream &OS, llvm::RecordKeeper &Records) {}

const char PublicAPICommand::Name[] = "public_api";

void PublicAPICommand::run(llvm::raw_ostream &OS, const ArgVector &Args,
                           llvm::StringRef StdHeader,
                           llvm::RecordKeeper &Records,
                           const Command::ErrorReporter &Reporter) const {
  if (Args.size() > 1) {
    Reporter.printFatalError(
        "public_api command does not take more than one arguments.");
  }

  AttributeStyle PreferedStyle = AttributeStyle::Gnu;

  for (auto &arg : Args) {
    if (arg == "prefer-c23-attributes") {
      PreferedStyle = AttributeStyle::C23;
    }
    if (arg == "prefer-no-attributes") {
      PreferedStyle = AttributeStyle::None;
    }
  }

  APIIndexer G(StdHeader, Records);
  writeAPIFromIndex(G, EntrypointNameList, OS, PreferedStyle);
}

} // namespace llvm_libc
