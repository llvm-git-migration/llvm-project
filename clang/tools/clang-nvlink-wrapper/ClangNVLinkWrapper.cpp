//===-- clang-nvlink-wrapper/ClangNVLinkWrapper.cpp - NVIDIA linker util --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This tool wraps around the NVIDIA linker called 'nvlink'. The NVIDIA linker
// is required to create NVPTX applications, but does not support common
// features like LTO or archives. This utility wraps around the tool to cover
// its deficiencies. This tool can be removed once NVIDIA improves their linker
// or ports it to `ld.lld`.
//
//===---------------------------------------------------------------------===//

#include "clang/Basic/Version.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace llvm::opt;
using namespace llvm::object;

static void printVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-nvlink-wrapper") << '\n';
}

/// The value of `argv[0]` when run.
static const char *Executable;

/// Temporary files to be cleaned up.
static SmallVector<SmallString<128>> TempFiles;

/// Codegen flags for LTO backend.
static codegen::RegisterCodeGenFlags CodeGenFlags;

namespace {
/// Must not overlap with llvm::opt::DriverFlag.
enum WrapperFlags {
  WrapperOnlyOption = (1 << 4), // Options only used by the linker wrapper.
  DeviceOnlyOption = (1 << 5),  // Options only used for device linking.
};

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "NVLinkOpts.inc"
  LastOption
#undef OPTION
};

#define PREFIX(NAME, VALUE)                                                    \
  static constexpr StringLiteral NAME##_init[] = VALUE;                        \
  static constexpr ArrayRef<StringLiteral> NAME(NAME##_init,                   \
                                                std::size(NAME##_init) - 1);
#include "NVLinkOpts.inc"
#undef PREFIX

static constexpr OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "NVLinkOpts.inc"
#undef OPTION
};

class WrapperOptTable : public opt::GenericOptTable {
public:
  WrapperOptTable() : opt::GenericOptTable(InfoTable) {}
};

const OptTable &getOptTable() {
  static const WrapperOptTable *Table = []() {
    auto Result = std::make_unique<WrapperOptTable>();
    return Result.release();
  }();
  return *Table;
}

[[noreturn]] void reportError(Error E) {
  outs().flush();
  logAllUnhandledErrors(std::move(E), WithColor::error(errs(), Executable));
  exit(EXIT_FAILURE);
}

void diagnosticHandler(const DiagnosticInfo &DI) {
  std::string ErrStorage;
  raw_string_ostream OS(ErrStorage);
  DiagnosticPrinterRawOStream DP(OS);
  DI.print(DP);

  switch (DI.getSeverity()) {
  case DS_Error:
    WithColor::error(errs(), Executable) << ErrStorage << "\n";
    break;
  case DS_Warning:
    WithColor::warning(errs(), Executable) << ErrStorage << "\n";
    break;
  case DS_Note:
    WithColor::note(errs(), Executable) << ErrStorage << "\n";
    break;
  case DS_Remark:
    WithColor::remark(errs()) << ErrStorage << "\n";
    break;
  }
}

Expected<StringRef> createTempFile(const ArgList &Args, const Twine &Prefix,
                                   StringRef Extension) {
  SmallString<128> OutputFile;
  if (Args.hasArg(OPT_save_temps)) {
    (Prefix + "." + Extension).toNullTerminatedStringRef(OutputFile);
  } else {
    if (std::error_code EC =
            sys::fs::createTemporaryFile(Prefix, Extension, OutputFile))
      return createFileError(OutputFile, EC);
  }

  TempFiles.emplace_back(std::move(OutputFile));
  return TempFiles.back();
}

Expected<std::string> findProgram(StringRef Name, ArrayRef<StringRef> Paths) {
  ErrorOr<std::string> Path = sys::findProgramByName(Name, Paths);
  if (!Path)
    Path = sys::findProgramByName(Name);
  if (!Path)
    return createStringError(Path.getError(),
                             "Unable to find '" + Name + "' in path");
  return *Path;
}

std::optional<std::string> findFile(StringRef Dir, StringRef Root,
                                    const Twine &Name) {
  SmallString<128> Path;
  if (Dir.starts_with("="))
    sys::path::append(Path, Root, Dir.substr(1), Name);
  else
    sys::path::append(Path, Dir, Name);

  if (sys::fs::exists(Path))
    return static_cast<std::string>(Path);
  return std::nullopt;
}

std::optional<std::string>
findFromSearchPaths(StringRef Name, StringRef Root,
                    ArrayRef<StringRef> SearchPaths) {
  for (StringRef Dir : SearchPaths)
    if (std::optional<std::string> File = findFile(Dir, Root, Name))
      return File;
  return std::nullopt;
}

std::optional<std::string>
searchLibraryBaseName(StringRef Name, StringRef Root,
                      ArrayRef<StringRef> SearchPaths) {
  for (StringRef Dir : SearchPaths)
    if (std::optional<std::string> File =
            findFile(Dir, Root, "lib" + Name + ".a"))
      return File;
  return std::nullopt;
}

/// Search for static libraries in the linker's library path given input like
/// `-lfoo` or `-l:libfoo.a`.
std::optional<std::string> searchLibrary(StringRef Input, StringRef Root,
                                         ArrayRef<StringRef> SearchPaths) {
  if (Input.starts_with(":") || Input.ends_with(".lib"))
    return findFromSearchPaths(Input.drop_front(), Root, SearchPaths);
  return searchLibraryBaseName(Input, Root, SearchPaths);
}

void printCommands(ArrayRef<StringRef> CmdArgs) {
  if (CmdArgs.empty())
    return;

  llvm::errs() << " \"" << CmdArgs.front() << "\" ";
  for (auto IC = std::next(CmdArgs.begin()), IE = CmdArgs.end(); IC != IE; ++IC)
    llvm::errs() << *IC << (std::next(IC) != IE ? " " : "\n");
}

/// A minimum symbol interface that provides the necessary information to
/// extract archive members and resolve LTO symbols.
struct Symbol {
  enum Flags {
    None = 0,
    Undefined = 1 << 0,
    Weak = 1 << 1,
  };

  Symbol()
      : File(), Flags(Undefined), Name(), UsedInRegularObj(false), Lazy(false) {
  }

  Symbol(MemoryBufferRef File, const irsymtab::Reader::SymbolRef Sym, bool Lazy)
      : File(File), Flags(0), UsedInRegularObj(false), Lazy(Lazy) {
    if (Sym.isUndefined())
      Flags |= Undefined;
    if (Sym.isWeak())
      Flags |= Weak;
    Name = Sym.getName();
  }

  Symbol(MemoryBufferRef File, const SymbolRef Sym, bool Lazy)
      : File(File), Flags(0), UsedInRegularObj(false), Lazy(Lazy) {
    auto FlagsOrErr = Sym.getFlags();
    if (!FlagsOrErr)
      reportError(FlagsOrErr.takeError());
    if (*FlagsOrErr & SymbolRef::SF_Undefined)
      Flags |= Undefined;
    if (*FlagsOrErr & SymbolRef::SF_Weak)
      Flags |= Weak;

    auto NameOrErr = Sym.getName();
    if (!NameOrErr)
      reportError(NameOrErr.takeError());
    Name = *NameOrErr;
  }

  Symbol Resolve(Symbol Other) {
    if (File.getBuffer().empty())
      return Other.Lazy ? *this : Other;
    if (Other.isUndefined())
      return *this;
    if (isWeak() && isUndefined() && Other.Lazy)
      return *this;
    if (isWeak() && !Other.isWeak())
      return Other;
    if (isUndefined() && !Other.isUndefined())
      return Other;
    return *this;
  }

  bool isWeak() const { return Flags & Weak; }
  bool isUndefined() const { return Flags & Undefined; }

  MemoryBufferRef File;
  uint32_t Flags;
  StringRef Name;
  bool UsedInRegularObj;
  bool Lazy;
};

Expected<StringRef> runPTXAs(StringRef File, const ArgList &Args) {
  std::string CudaPath = Args.getLastArgValue(OPT_cuda_path_EQ).str();
  Expected<std::string> PTXAsPath = findProgram("ptxas", {CudaPath + "/bin"});
  if (!PTXAsPath)
    return PTXAsPath.takeError();

  auto TempFileOrErr = createTempFile(
      Args, sys::path::stem(Args.getLastArgValue(OPT_o, "a.out")), "cubin");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  SmallVector<StringRef> AssemblerArgs({*PTXAsPath, "-m64", "-c", File});
  if (Args.hasArg(OPT_g))
    AssemblerArgs.push_back("-O0");
  else
    AssemblerArgs.push_back(
        Args.MakeArgString("-O" + Args.getLastArgValue(OPT_O, "3")));
  AssemblerArgs.append({"-arch", Args.getLastArgValue(OPT_arch)});
  AssemblerArgs.append({"-o", *TempFileOrErr});

  if (Args.hasArg(OPT_dry_run))
    printCommands(AssemblerArgs);
  else if (sys::ExecuteAndWait(*PTXAsPath, AssemblerArgs))
    return createStringError("'" + sys::path::filename(*PTXAsPath) + "'" +
                             " failed");
  return Args.MakeArgString(*TempFileOrErr);
}

Expected<std::unique_ptr<lto::LTO>> createLTO(const ArgList &Args) {
  const llvm::Triple Triple("nvptx64-nvidia-cuda");
  // We need to remove AMD's target-id from the processor if present.
  lto::Config Conf;
  lto::ThinBackend Backend;
  Backend =
      lto::createInProcessThinBackend(llvm::heavyweight_hardware_concurrency());

  Conf.CPU = Args.getLastArgValue(OPT_arch);
  Conf.Options = codegen::InitTargetOptionsFromCodeGenFlags(Triple);

  Conf.MAttrs = {Args.getLastArgValue(OPT_feature, "+ptx60").str()};
  std::optional<CodeGenOptLevel> CGOptLevelOrNone =
      CodeGenOpt::parseLevel(Args.getLastArgValue(OPT_O, "2")[0]);
  assert(CGOptLevelOrNone && "Invalid optimization level");
  Conf.CGOptLevel = *CGOptLevelOrNone;
  Conf.OptLevel = Args.getLastArgValue(OPT_O, "2")[0] - '0';
  Conf.DefaultTriple = Triple.getTriple();

  Conf.DiagHandler = diagnosticHandler;
  Conf.CGFileType = CodeGenFileType::AssemblyFile;

  if (Args.hasArg(OPT_save_temps))
    if (Error Err = Conf.addSaveTemps(
            (Args.getLastArgValue(OPT_o, "a.out") + ".").str()))
      return Err;

  return std::make_unique<lto::LTO>(std::move(Conf), Backend);
}

Expected<SmallVector<StringRef>> getInput(const ArgList &Args) {
  SmallVector<StringRef> LibraryPaths;
  for (const opt::Arg *Arg : Args.filtered(OPT_library_path))
    LibraryPaths.push_back(Arg->getValue());

  bool WholeArchive = false;
  SmallVector<std::pair<std::unique_ptr<MemoryBuffer>, bool>> InputFiles;
  for (const opt::Arg *Arg : Args.filtered(
           OPT_INPUT, OPT_library, OPT_whole_archive, OPT_no_whole_archive)) {
    if (Arg->getOption().matches(OPT_whole_archive) ||
        Arg->getOption().matches(OPT_no_whole_archive)) {
      WholeArchive = Arg->getOption().matches(OPT_whole_archive);
      continue;
    }

    std::optional<std::string> Filename =
        Arg->getOption().matches(OPT_library)
            ? searchLibrary(Arg->getValue(), /*Root=*/"", LibraryPaths)
            : std::string(Arg->getValue());

    if (!Filename && Arg->getOption().matches(OPT_library))
      return createStringError("unable to find library -l%s", Arg->getValue());

    if (!Filename || !sys::fs::exists(*Filename) ||
        sys::fs::is_directory(*Filename))
      continue;

    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getFileOrSTDIN(*Filename);
    if (std::error_code EC = BufferOrErr.getError())
      return createFileError(*Filename, EC);

    MemoryBufferRef Buffer = **BufferOrErr;
    switch (identify_magic(Buffer.getBuffer())) {
    case file_magic::bitcode:
    case file_magic::elf_relocatable:
      InputFiles.emplace_back(std::move(*BufferOrErr), /*IsLazy=*/false);
      break;
    case file_magic::archive: {
      Expected<std::unique_ptr<llvm::object::Archive>> LibFile =
          object::Archive::create(Buffer);
      if (!LibFile)
        return LibFile.takeError();
      Error Err = Error::success();
      for (auto Child : (*LibFile)->children(Err)) {
        auto ChildBufferOrErr = Child.getMemoryBufferRef();
        if (!ChildBufferOrErr)
          return ChildBufferOrErr.takeError();
        std::unique_ptr<MemoryBuffer> ChildBuffer =
            MemoryBuffer::getMemBufferCopy(
                ChildBufferOrErr->getBuffer(),
                ChildBufferOrErr->getBufferIdentifier());
        InputFiles.emplace_back(std::move(ChildBuffer), !WholeArchive);
      }
      if (Err)
        return Err;
      break;
    }
    default:
      return createStringError("Unsupported file type");
    }
  }

  StringMap<Symbol> SymTab;
  SmallVector<std::unique_ptr<MemoryBuffer>> LinkerInput;

  bool Extracted = true;
  while (Extracted) {
    Extracted = false;
    for (auto &[Buffer, IsLazy] : InputFiles) {
      if (!Buffer)
        continue;

      Extracted = !IsLazy;
      MemoryBufferRef File = *Buffer;
      switch (identify_magic(File.getBuffer())) {
      case file_magic::bitcode: {
        Expected<IRSymtabFile> IRSymtabOrErr = readIRSymtab(File);
        if (!IRSymtabOrErr)
          return IRSymtabOrErr.takeError();
        bool CheckSymbolsAgain;
        do {
          CheckSymbolsAgain = false;
          for (unsigned I = 0; I != IRSymtabOrErr->Mods.size(); ++I) {
            for (const auto &IRSym :
                 IRSymtabOrErr->TheReader.module_symbols(I)) {
              if (IRSym.isFormatSpecific() || !IRSym.isGlobal())
                continue;

              Symbol &Sym = SymTab[IRSym.getName()];
              Sym = Sym.Resolve(Symbol(File, IRSym, IsLazy));
              if (Sym.File == File) {
                Extracted = true;
                if (IsLazy) {
                  IsLazy = false;
                  CheckSymbolsAgain = true;
                }
              }
            }
          }
        } while (CheckSymbolsAgain);
        break;
      }
      case file_magic::elf_relocatable: {
        Expected<std::unique_ptr<ObjectFile>> ObjFile =
            ObjectFile::createObjectFile(File);
        if (!ObjFile)
          return ObjFile.takeError();
        bool CheckSymbolsAgaing;
        do {
          CheckSymbolsAgaing = false;
          for (SymbolRef ObjSum : (*ObjFile)->symbols()) {
            auto NameOrErr = ObjSum.getName();
            if (!NameOrErr)
              return NameOrErr.takeError();

            Symbol &Sym = SymTab[*NameOrErr];
            Sym = Sym.Resolve(Symbol(File, ObjSum, IsLazy));
            Sym.UsedInRegularObj = true;
            if (Sym.File == File) {
              Extracted = true;
              if (IsLazy) {
                IsLazy = false;
                CheckSymbolsAgaing = true;
              }
            }
          }
        } while (CheckSymbolsAgaing);
        break;
      }
      default:
        return createStringError("Unsupported file type");
      }

      // If we extracted any files we need to check all the symbols again.
      if (Extracted) {
        LinkerInput.emplace_back(std::move(Buffer));
        break;
      }
    }
  }

  SmallVector<std::unique_ptr<MemoryBuffer>> BitcodeFiles;
  for (auto &Input : LinkerInput)
    if (identify_magic(Input->getBuffer()) == file_magic::bitcode)
      BitcodeFiles.emplace_back(std::move(Input));

  SmallVector<StringRef> Files;
  if (!BitcodeFiles.empty()) {
    auto LTOBackendOrErr = createLTO(Args);
    if (!LTOBackendOrErr)
      return LTOBackendOrErr.takeError();
    lto::LTO &LTOBackend = **LTOBackendOrErr;
    for (auto &BitcodeFile : BitcodeFiles) {
      Expected<std::unique_ptr<lto::InputFile>> BitcodeFileOrErr =
          llvm::lto::InputFile::create(*BitcodeFile);
      if (!BitcodeFileOrErr)
        return BitcodeFileOrErr.takeError();

      const auto Symbols = (*BitcodeFileOrErr)->symbols();
      SmallVector<lto::SymbolResolution, 16> Resolutions(Symbols.size());
      size_t Idx = 0;
      for (auto &Sym : Symbols) {
        lto::SymbolResolution &Res = Resolutions[Idx++];
        Symbol ObjSym = SymTab[Sym.getName()];

        // We will use this as the prevailing symbol in LTO if it is not
        // undefined and it is from the file that contained the canonical
        // definition.
        Res.Prevailing = !Sym.isUndefined() && ObjSym.File == *BitcodeFile;

        // We need LTO to preseve the following global symbols:
        // 1) Symbols used in regular objects.
        // 2) Prevailing symbols that are needed visible to the gpu runtime.
        Res.VisibleToRegularObj =
            ObjSym.UsedInRegularObj ||
            (Res.Prevailing &&
             (Sym.getVisibility() != GlobalValue::HiddenVisibility &&
              !Sym.canBeOmittedFromSymbolTable()));

        // Identify symbols that must be exported dynamically and can be
        // referenced by other files, (i.e. the runtime).
        Res.ExportDynamic =
            Sym.getVisibility() != GlobalValue::HiddenVisibility &&
            !Sym.canBeOmittedFromSymbolTable();

        // The final definition will reside in this linkage unit if the symbol
        // is defined and local to the module. This only checks for bitcode
        // files, full assertion will require complete symbol resolution.
        Res.FinalDefinitionInLinkageUnit =
            Sym.getVisibility() != GlobalValue::DefaultVisibility &&
            (!Sym.isUndefined() && !Sym.isCommon());

        // We do not support linker redefined symbols (e.g. --wrap) for device
        // image linking, so the symbols will not be changed after LTO.
        Res.LinkerRedefined = false;
      }

      // Add the bitcode file with its resolved symbols to the LTO job.
      if (Error Err = LTOBackend.add(std::move(*BitcodeFileOrErr), Resolutions))
        return Err;
    }

    // Run the LTO job to compile the bitcode.
    size_t MaxTasks = LTOBackend.getMaxTasks();
    SmallVector<StringRef> LTOFiles(MaxTasks);
    auto AddStream =
        [&](size_t Task,
            const Twine &ModuleName) -> std::unique_ptr<CachedFileStream> {
      int FD = -1;
      auto &TempFile = LTOFiles[Task];
      auto TempFileOrErr = createTempFile(
          Args, sys::path::stem(Args.getLastArgValue(OPT_o, "a.out")), "s");
      if (!TempFileOrErr)
        reportError(TempFileOrErr.takeError());
      TempFile = Args.MakeArgString(*TempFileOrErr);
      if (std::error_code EC = sys::fs::openFileForWrite(TempFile, FD))
        reportError(errorCodeToError(EC));
      return std::make_unique<CachedFileStream>(
          std::make_unique<llvm::raw_fd_ostream>(FD, true));
    };

    if (Error Err = LTOBackend.run(AddStream))
      return Err;

    for (StringRef LTOFile : LTOFiles) {
      auto FileOrErr = runPTXAs(LTOFile, Args);
      if (!FileOrErr)
        return FileOrErr.takeError();
      Files.emplace_back(*FileOrErr);
    }
  }

  for (auto &Input : LinkerInput) {
    if (!Input)
      continue;

    auto TempFileOrErr = createTempFile(
        Args, sys::path::stem(Input->getBufferIdentifier()), "cubin");
    if (!TempFileOrErr)
      return TempFileOrErr.takeError();
    Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
        FileOutputBuffer::create(*TempFileOrErr, Input->getBuffer().size());
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
    llvm::copy(Input->getBuffer(), Output->getBufferStart());
    if (Error E = Output->commit())
      return E;
    Files.emplace_back(Args.MakeArgString(*TempFileOrErr));
  }

  return Files;
}

Error runNVLink(ArrayRef<StringRef> Files, const ArgList &Args) {
  std::string CudaPath = Args.getLastArgValue(OPT_cuda_path_EQ).str();
  Expected<std::string> NVLinkPath = findProgram("nvlink", {CudaPath + "/bin"});
  if (!NVLinkPath)
    return NVLinkPath.takeError();

  ArgStringList NewLinkerArgs;
  for (const opt::Arg *Arg : Args) {
    // Do not forward arguments only intended for the linker wrapper.
    if (Arg->getOption().hasFlag(WrapperOnlyOption))
      continue;

    // Do not forward any inputs that we have processed.
    if (Arg->getOption().matches(OPT_INPUT) ||
        Arg->getOption().matches(OPT_library))
      continue;

    Arg->render(Args, NewLinkerArgs);
  }

  llvm::transform(Files, std::back_inserter(NewLinkerArgs),
                  [&](StringRef Arg) { return Args.MakeArgString(Arg); });

  SmallVector<StringRef> LinkerArgs({*NVLinkPath});
  if (!Args.hasArg(OPT_o))
    LinkerArgs.append({"-o", "a.out"});
  for (StringRef Arg : NewLinkerArgs)
    LinkerArgs.push_back(Arg);

  if (Args.hasArg(OPT_dry_run))
    printCommands(LinkerArgs);
  else if (sys::ExecuteAndWait(*NVLinkPath, LinkerArgs))
    return createStringError("'" + sys::path::filename(*NVLinkPath) + "'" +
                             " failed");
  return Error::success();
}

} // namespace

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  Executable = argv[0];
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  const OptTable &Tbl = getOptTable();
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  auto Args = Tbl.parseArgs(argc, argv, OPT_INVALID, Saver, [&](StringRef Err) {
    reportError(createStringError(inconvertibleErrorCode(), Err));
  });

  if (Args.hasArg(OPT_help) || Args.hasArg(OPT_help_hidden)) {
    Tbl.printHelp(
        outs(), "clang-nvlink-wrapper [options] <options to passed to nvlink>",
        "A utility that wraps around the NVIDIA 'nvlink' linker.\n"
        "This enables static linking and LTO handling for NVPTX targets.\n",
        Args.hasArg(OPT_help_hidden), Args.hasArg(OPT_help_hidden));
    return EXIT_SUCCESS;
  }

  if (Args.hasArg(OPT_version))
    printVersion(outs());

  // This forwards '-mllvm' arguments to LLVM if present.
  SmallVector<const char *> NewArgv = {argv[0]};
  for (const opt::Arg *Arg : Args.filtered(OPT_mllvm))
    NewArgv.push_back(Arg->getValue());
  cl::ParseCommandLineOptions(NewArgv.size(), &NewArgv[0]);

  auto FilesOrErr = getInput(Args);
  if (!FilesOrErr)
    reportError(FilesOrErr.takeError());

  if (Error Err = runNVLink(*FilesOrErr, Args))
    reportError(std::move(Err));

  // Remove the temporary files created.
  if (!Args.hasArg(OPT_save_temps))
    for (const auto &TempFile : TempFiles)
      if (std::error_code EC = sys::fs::remove(TempFile))
        reportError(createFileError(TempFile, EC));

  return EXIT_SUCCESS;
}
