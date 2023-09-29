//===- lib/MC/GOFFObjectWriter.cpp - GOFF File Writer ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements GOFF object file writer information.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCGOFFObjectWriter.h"
#include "llvm/MC/MCSectionGOFF.h"
#include "llvm/MC/MCSymbolGOFF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ConvertEBCDIC.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "goff-writer"

namespace {

// The standard System/390 convention is to name the high-order (leftmost) bit
// in a byte as bit zero. The Flags type helps to set bits in a byte according
// to this numeration order.
class Flags {
  uint8_t Val;

  constexpr static uint8_t bits(uint8_t BitIndex, uint8_t Length, uint8_t Value,
                                uint8_t OldValue) {
    assert(BitIndex < 8 && "Bit index out of bounds!");
    assert(Length + BitIndex <= 8 && "Bit length too long!");

    uint8_t Mask = ((1 << Length) - 1) << (8 - BitIndex - Length);
    Value = Value << (8 - BitIndex - Length);
    assert((Value & Mask) == Value && "Bits set outside of range!");

    return (OldValue & ~Mask) | Value;
  }

public:
  constexpr Flags() : Val(0) {}
  constexpr Flags(uint8_t BitIndex, uint8_t Length, uint8_t Value)
      : Val(bits(BitIndex, Length, Value, 0)) {}

  void set(uint8_t BitIndex, uint8_t Length, uint8_t Value) {
    Val = bits(BitIndex, Length, Value, Val);
  }

  constexpr operator uint8_t() const { return Val; }
};

// Common flag values on records.

// Flag: This record is continued.
constexpr uint8_t RecContinued = Flags(7, 1, 1);

// Flag: This record is a continuation.
constexpr uint8_t RecContinuation = Flags(6, 1, 1);

// The GOFFOstream is responsible to write the data into the fixed physical
// records of the format. A user of this class announces the start of a new
// logical record and the size of its content. While writing the content, the
// physical records are created for the data. Possible fill bytes at the end of
// a physical record are written automatically. In principle, the GOFFOstream
// is agnostic of the endianness of the content. However, it also supports
// writing data in big endian byte order.
class GOFFOstream : public raw_ostream {
  /// The underlying raw_pwrite_stream.
  raw_pwrite_stream &OS;

  /// The remaining size of this logical record, including fill bytes.
  size_t RemainingSize;

#ifndef NDEBUG
  /// The number of bytes needed to fill up the last physical record.
  size_t Gap = 0;
#endif

  /// The number of logical records emitted to far.
  uint32_t LogicalRecords;

  /// The type of the current (logical) record.
  GOFF::RecordType CurrentType;

  /// Signals start of new record.
  bool NewLogicalRecord;

  /// Static allocated buffer for the stream, used by the raw_ostream class. The
  /// buffer is sized to hold the content of a physical record.
  char Buffer[GOFF::RecordContentLength];

  // Return the number of bytes left to write until next physical record.
  // Please note that we maintain the total numbers of byte left, not the
  // written size.
  size_t bytesToNextPhysicalRecord() {
    size_t Bytes = RemainingSize % GOFF::RecordContentLength;
    return Bytes ? Bytes : GOFF::RecordContentLength;
  }

  /// Write the record prefix of a physical record, using the given record type.
  static void writeRecordPrefix(raw_ostream &OS, GOFF::RecordType Type,
                                size_t RemainingSize,
                                uint8_t Flags = RecContinuation);

  /// Fill the last physical record of a logical record with zero bytes.
  void fillRecord();

  /// See raw_ostream::write_impl.
  void write_impl(const char *Ptr, size_t Size) override;

  /// Return the current position within the stream, not counting the bytes
  /// currently in the buffer.
  uint64_t current_pos() const override { return OS.tell(); }

public:
  explicit GOFFOstream(raw_pwrite_stream &OS)
      : OS(OS), RemainingSize(0), LogicalRecords(0), NewLogicalRecord(false) {
    SetBuffer(Buffer, sizeof(Buffer));
  }

  ~GOFFOstream() { finalize(); }

  raw_pwrite_stream &getOS() { return OS; }

  void newRecord(GOFF::RecordType Type, size_t Size);

  void finalize() { fillRecord(); }

  uint32_t logicalRecords() { return LogicalRecords; }

  // Support for endian-specific data.
  template <typename value_type> void writebe(value_type Value) {
    Value =
        support::endian::byte_swap<value_type>(Value, llvm::endianness::big);
    write(reinterpret_cast<const char *>(&Value), sizeof(value_type));
  }
};

void GOFFOstream::writeRecordPrefix(raw_ostream &OS, GOFF::RecordType Type,
                                    size_t RemainingSize, uint8_t Flags) {
  uint8_t TypeAndFlags = Flags | (Type << 4);
  if (RemainingSize > GOFF::RecordLength)
    TypeAndFlags |= RecContinued;
  OS << static_cast<unsigned char>(GOFF::PTVPrefix) // Record Type
     << static_cast<unsigned char>(TypeAndFlags)    // Continuation
     << static_cast<unsigned char>(0);              // Version
}

void GOFFOstream::newRecord(GOFF::RecordType Type, size_t Size) {
  fillRecord();
  CurrentType = Type;
  RemainingSize = Size;
#ifdef NDEBUG
  size_t Gap;
#endif
  Gap = (RemainingSize % GOFF::RecordContentLength);
  if (Gap) {
    Gap = GOFF::RecordContentLength - Gap;
    RemainingSize += Gap;
  }
  NewLogicalRecord = true;
  ++LogicalRecords;
}

void GOFFOstream::fillRecord() {
  assert((GetNumBytesInBuffer() <= RemainingSize) &&
         "More bytes in buffer than expected");
  size_t Remains = RemainingSize - GetNumBytesInBuffer();
  if (Remains) {
    assert(Remains == Gap && "Wrong size of fill gap");
    assert((Remains < GOFF::RecordLength) &&
           "Attempt to fill more than one physical record");
    raw_ostream::write_zeros(Remains);
  }
  flush();
  assert(RemainingSize == 0 && "Not fully flushed");
  assert(GetNumBytesInBuffer() == 0 && "Buffer not fully empty");
}

// This function is called from the raw_ostream implementation if:
// - The internal buffer is full. Size is excactly the size of the buffer.
// - Data larger than the internal buffer is written. Size is a multiple of the
//   buffer size.
// - flush() has been called. Size is at most the buffer size.
// The GOFFOstream implementation ensures that flush() is called before a new
// logical record begins. Therefore it is sufficient to check for a new block
// only once.
void GOFFOstream::write_impl(const char *Ptr, size_t Size) {
  assert((RemainingSize >= Size) && "Attempt to write too much data");
  assert(RemainingSize && "Logical record overflow");
  if (!(RemainingSize % GOFF::RecordContentLength)) {
    writeRecordPrefix(OS, CurrentType, RemainingSize,
                      NewLogicalRecord ? 0 : RecContinuation);
    NewLogicalRecord = false;
  }
  assert(!NewLogicalRecord &&
         "New logical record not on physical record boundary");

  size_t Idx = 0;
  while (Size > 0) {
    size_t BytesToWrite = bytesToNextPhysicalRecord();
    if (BytesToWrite > Size)
      BytesToWrite = Size;
    OS.write(Ptr + Idx, BytesToWrite);
    Idx += BytesToWrite;
    Size -= BytesToWrite;
    RemainingSize -= BytesToWrite;
    if (Size)
      writeRecordPrefix(OS, CurrentType, RemainingSize);
  }
}

/// \brief Wrapper class for symbols used exclusively for the symbol table in a
/// GOFF file.
class GOFFSymbol {
public:
  std::string Name;
  uint32_t EsdId;
  uint32_t ParentEsdId;
  const MCSymbolGOFF *MCSym;
  GOFF::ESDSymbolType SymbolType;

  GOFF::ESDNameSpaceId NameSpace = GOFF::ESD_NS_NormalName;
  GOFF::ESDAmode Amode = GOFF::ESD_AMODE_64;
  GOFF::ESDRmode Rmode = GOFF::ESD_RMODE_64;
  GOFF::ESDLinkageType Linkage = GOFF::ESD_LT_XPLink;
  GOFF::ESDExecutable Executable = GOFF::ESD_EXE_Unspecified;
  GOFF::ESDAlignment Alignment = GOFF::ESD_ALIGN_Byte;
  GOFF::ESDTextStyle TextStyle = GOFF::ESD_TS_ByteOriented;
  GOFF::ESDBindingAlgorithm BindAlgorithm = GOFF::ESD_BA_Concatenate;
  GOFF::ESDLoadingBehavior LoadBehavior = GOFF::ESD_LB_Initial;
  GOFF::ESDBindingScope BindingScope = GOFF::ESD_BSC_Unspecified;
  GOFF::ESDBindingStrength BindingStrength = GOFF::ESD_BST_Strong;
  uint32_t SortKey = 0;
  uint32_t SectionLength = 0;
  uint32_t ADAEsdId = 0;
  bool Indirect = false;
  bool ForceRent = false;
  bool Renamable = false;
  bool ReadOnly = false;
  uint32_t EASectionEsdId = 0;
  uint32_t EASectionOffset = 0;

  GOFFSymbol(StringRef Name, GOFF::ESDSymbolType Type, uint32_t EsdID,
             uint32_t ParentEsdID)
      : Name(Name.data(), Name.size()), EsdId(EsdID), ParentEsdId(ParentEsdID),
        MCSym(nullptr), SymbolType(Type) {}

  bool isForceRent() const { return ForceRent; }
  bool isReadOnly() const { return ReadOnly; }
  bool isRemovable() const { return false; }
  bool isExecutable() const { return Executable == GOFF::ESD_EXE_CODE; }
  bool isExecUnspecified() const {
    return Executable == GOFF::ESD_EXE_Unspecified;
  }
  bool isWeakRef() const { return BindingStrength == GOFF::ESD_BST_Weak; }
  bool isExternal() const {
    return (BindingScope == GOFF::ESD_BSC_Library) ||
           (BindingScope == GOFF::ESD_BSC_ImportExport);
  }

  void setAlignment(Align A) {
    // The GOFF alignment is encoded as log_2 value.
    uint8_t Log = Log2(A);
    if (Log <= GOFF::ESD_ALIGN_4Kpage)
      Alignment = static_cast<GOFF::ESDAlignment>(Log);
    else
      llvm_unreachable("Unsupported alignment");
  }

  void setMaxAlignment(Align A) {
    GOFF::ESDAlignment CurrAlign = Alignment;
    setAlignment(A);
    if (CurrAlign > Alignment)
      Alignment = CurrAlign;
  }
};

/// \brief Wrapper class for sections used exclusively for representing sections
/// of the GOFF output that have actual bytes.  This could be a ED or a PR.
/// Relocations will always have a P-pointer to the ESDID of one of these.
class GOFFSection {
public:
  GOFFSymbol *Pptr = nullptr;
  GOFFSymbol *Rptr = nullptr;
  GOFFSymbol *SD = nullptr;
  const MCSectionGOFF *MCSec = nullptr;
  bool IsStructured = false;

  GOFFSection(GOFFSymbol *Pptr, GOFFSymbol *Rptr, GOFFSymbol *SD,
              const MCSectionGOFF *MCSec)
      : Pptr(Pptr), Rptr(Rptr), SD(SD), MCSec(MCSec), IsStructured(false) {}
};

class GOFFObjectWriter : public MCObjectWriter {
  typedef std::vector<std::unique_ptr<GOFFSymbol>> SymbolListType;
  typedef DenseMap<MCSymbol const *, GOFFSymbol *> SymbolMapType;
  typedef std::vector<std::unique_ptr<GOFFSection>> SectionListType;
  typedef DenseMap<MCSection const *, GOFFSection *> SectionMapType;

  // The target specific GOFF writer instance.
  std::unique_ptr<MCGOFFObjectTargetWriter> TargetObjectWriter;

  /// The symbol table for a GOFF file.  It is order sensitive.
  SymbolListType EsdSymbols;

  /// Lookup table for MCSymbols to GOFFSymbols.  Needed to determine EsdIds
  /// of symbols in Relocations.
  SymbolMapType SymbolMap;

  /// The list of sections for the GOFF file.
  SectionListType Sections;

  /// Lookup table for MCSections to GOFFSections.  Needed to determine
  /// SymbolType on GOFFSymbols that reside in GOFFSections.
  SectionMapType SectionMap;

  // The stream used to write the GOFF records.
  GOFFOstream OS;

  uint32_t EsdCounter = 1;

  GOFFSymbol *RootSD = nullptr;
  GOFFSymbol *CodeLD = nullptr;
  GOFFSymbol *ADA = nullptr;
  bool HasADA = false;

public:
  GOFFObjectWriter(std::unique_ptr<MCGOFFObjectTargetWriter> MOTW,
                   raw_pwrite_stream &OS)
      : TargetObjectWriter(std::move(MOTW)), OS(OS) {}

  ~GOFFObjectWriter() override {}

private:
  // Write GOFF records.
  void writeHeader();

  void writeSymbol(const GOFFSymbol &Symbol, const MCAsmLayout &Layout);
  void writeText(const GOFFSection &Section, const MCAssembler &Asm,
                 const MCAsmLayout &Layout);

  void writeEnd();

public:
  // Implementation of the MCObjectWriter interface.
  void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) override {}
  void executePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override;
  uint64_t writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;

private:
  GOFFSection *createGOFFSection(GOFFSymbol *Pptr, GOFFSymbol *Rptr,
                                 GOFFSymbol *SD, const MCSectionGOFF *MC);
  GOFFSymbol *createGOFFSymbol(StringRef Name, GOFF::ESDSymbolType Type,
                               uint32_t ParentEsdId);
  GOFFSymbol *createSDSymbol(StringRef Name);
  GOFFSymbol *createEDSymbol(StringRef Name, uint32_t ParentEsdId);
  GOFFSymbol *createLDSymbol(StringRef Name, uint32_t ParentEsdId);
  GOFFSymbol *createERSymbol(StringRef Name, uint32_t ParentEsdId,
                             const MCSymbolGOFF *Source = nullptr);
  GOFFSymbol *createPRSymbol(StringRef Name, uint32_t ParentEsdId);

  GOFFSymbol *createWSASymbol(uint32_t ParentEsdId);
  void defineRootAndADASD(MCAssembler &Asm);
  void defineSectionSymbols(const MCAssembler &Asm,
                            const MCSectionGOFF &Section,
                            const MCAsmLayout &Layout);
  void processSymbolDefinedInModule(const MCSymbolGOFF &MCSymbol,
                                    const MCAssembler &Asm,
                                    const MCAsmLayout &Layout);
  void processSymbolDeclaredInModule(const MCSymbolGOFF &Symbol);
};
} // end anonymous namespace

GOFFSection *GOFFObjectWriter::createGOFFSection(GOFFSymbol *Pptr,
                                                 GOFFSymbol *Rptr,
                                                 GOFFSymbol *SD,
                                                 const MCSectionGOFF *MC) {
  Sections.push_back(std::make_unique<GOFFSection>(Pptr, Rptr, SD, MC));

  return Sections.back().get();
}

GOFFSymbol *GOFFObjectWriter::createGOFFSymbol(StringRef Name,
                                               GOFF::ESDSymbolType Type,
                                               uint32_t ParentEsdId) {
  EsdSymbols.push_back(
      std::make_unique<GOFFSymbol>(Name, Type, EsdCounter, ParentEsdId));
  ++EsdCounter;
  return EsdSymbols.back().get();
}

GOFFSymbol *GOFFObjectWriter::createSDSymbol(StringRef Name) {
  return createGOFFSymbol(Name, GOFF::ESD_ST_SectionDefinition, 0);
}

GOFFSymbol *GOFFObjectWriter::createEDSymbol(StringRef Name,
                                             uint32_t ParentEsdId) {
  GOFFSymbol *ED =
      createGOFFSymbol(Name, GOFF::ESD_ST_ElementDefinition, ParentEsdId);

  ED->Alignment = GOFF::ESD_ALIGN_Doubleword;
  return ED;
}

GOFFSymbol *GOFFObjectWriter::createLDSymbol(StringRef Name,
                                             uint32_t ParentEsdId) {
  return createGOFFSymbol(Name, GOFF::ESD_ST_LabelDefinition, ParentEsdId);
}

GOFFSymbol *GOFFObjectWriter::createERSymbol(StringRef Name,
                                             uint32_t ParentEsdId,
                                             const MCSymbolGOFF *Source) {
  GOFFSymbol *ER =
      createGOFFSymbol(Name, GOFF::ESD_ST_ExternalReference, ParentEsdId);

  if (Source) {
    ER->Linkage = GOFF::ESDLinkageType::ESD_LT_XPLink;
    ER->Executable = Source->getExecutable();
    ER->BindingScope = Source->isExternal()
                           ? GOFF::ESDBindingScope::ESD_BSC_Library
                           : GOFF::ESDBindingScope::ESD_BSC_Section;
    ER->BindingStrength = Source->isWeak()
                              ? GOFF::ESDBindingStrength::ESD_BST_Weak
                              : GOFF::ESDBindingStrength::ESD_BST_Strong;
  }

  return ER;
}

GOFFSymbol *GOFFObjectWriter::createPRSymbol(StringRef Name,
                                             uint32_t ParentEsdId) {
  return createGOFFSymbol(Name, GOFF::ESD_ST_PartReference, ParentEsdId);
}

GOFFSymbol *GOFFObjectWriter::createWSASymbol(uint32_t ParentEsdId) {
  const char *WSAClassName = "C_WSA64";
  GOFFSymbol *WSA = createEDSymbol(WSAClassName, ParentEsdId);

  WSA->Executable = GOFF::ESD_EXE_DATA;
  WSA->TextStyle = GOFF::ESD_TS_ByteOriented;
  WSA->BindAlgorithm = GOFF::ESD_BA_Merge;
  WSA->Alignment = GOFF::ESD_ALIGN_Quadword;
  WSA->LoadBehavior = GOFF::ESD_LB_Deferred;
  WSA->NameSpace = GOFF::ESD_NS_Parts;
  WSA->SectionLength = 0;

  return WSA;
}

void GOFFObjectWriter::defineRootAndADASD(MCAssembler &Asm) {
  assert(!RootSD && !ADA && "SD already initialzed");
  StringRef FileName = "";
  if (Asm.getFileNames().size())
    FileName = sys::path::stem((*(Asm.getFileNames().begin())).first);
  std::pair<std::string, std::string> CsectNames = Asm.getCsectNames();
  if (CsectNames.first.empty()) {
    RootSD = createSDSymbol(FileName.str().append("#C"));
    RootSD->BindingScope = GOFF::ESD_BSC_Section;
  } else {
    RootSD = createSDSymbol(CsectNames.first);
  }
  RootSD->Executable = GOFF::ESD_EXE_CODE;

  GOFFSymbol *ADAED = createWSASymbol(RootSD->EsdId);
  if (CsectNames.second.empty()) {
    ADA = createPRSymbol(FileName.str().append("#S"), ADAED->EsdId);
    ADA->BindingScope = GOFF::ESD_BSC_Section;
  } else {
    ADA = createPRSymbol(CsectNames.second, ADAED->EsdId);
    ADA->BindingScope = GOFF::ESD_BSC_Library;
  }
  ADA->Executable = GOFF::ESD_EXE_DATA;
  ADA->NameSpace = GOFF::ESD_NS_Parts;
  ADA->Alignment = GOFF::ESD_ALIGN_Quadword;
  // The ADA Section length can increase after this point. The final ADA
  // length is assigned in populateADASection().
  // We need to be careful, because a value of 0 causes issues with function
  // labels. In populateADASection(), we make sure that the ADA content is
  // defined before createing function labels.
  ADA->SectionLength = 0;
  // Assume there is an ADA.
  HasADA = true;
}

void GOFFObjectWriter::defineSectionSymbols(const MCAssembler &Asm,
                                            const MCSectionGOFF &Section,
                                            const MCAsmLayout &Layout) {
  GOFFSection *GSection = nullptr;
  SectionKind Kind = Section.getKind();

  if (Kind.isText()) {
    GOFFSymbol *SD = RootSD;
    const char *CodeClassName = "C_CODE64";
    GOFFSymbol *ED = createEDSymbol(CodeClassName, SD->EsdId);
    GOFFSymbol *LD = createLDSymbol(SD->Name, ED->EsdId);

    ED->SectionLength = Layout.getSectionAddressSize(&Section);
    ED->Executable = GOFF::ESD_EXE_CODE;
    ED->ForceRent = true;

    LD->Executable = GOFF::ESD_EXE_CODE;
    if (SD->BindingScope == GOFF::ESD_BSC_Section) {
      LD->BindingScope = GOFF::ESD_BSC_Section;
    } else {
      LD->BindingScope = GOFF::ESD_BSC_Library;
    }

    CodeLD = LD;

    GSection = createGOFFSection(ED, LD, SD, &Section);
  } else if (Section.getName().equals(".ada")) {
    // Symbols already defined in defineRootAndADASD, nothing to do.
    ADA->SectionLength = Layout.getSectionAddressSize(&Section);
    if (ADA->SectionLength)
      CodeLD->ADAEsdId = ADA->EsdId;
    else {
      // We cannot have a zero-length section for data.  If we do, artificially
      // inflate it.  Use 2 bytes to avoid odd alignments.
      ADA->SectionLength = 2;
      HasADA = false;
    }
    GSection = createGOFFSection(ADA, ADA, RootSD, &Section);
  } else if (Kind.isBSS() || Kind.isData()) {
    // Statics and globals that are defined.
    StringRef SectionName = Section.getName();
    GOFFSymbol *SD = createSDSymbol(SectionName);

    // Determine if static/global variable is marked with the norent attribute.
    MCContext &Ctx = Asm.getContext();
    auto *Sym = cast_or_null<MCSymbolGOFF>(Ctx.lookupSymbol(SectionName));

    if (Sym) {
      GOFFSymbol *ED = createWSASymbol(SD->EsdId);
      GOFFSymbol *PR = createPRSymbol(SectionName, ED->EsdId);
      ED->Alignment =
          std::max(static_cast<GOFF::ESDAlignment>(Log2(Section.getAlign())),
                   GOFF::ESD_ALIGN_Quadword);

      PR->Executable = GOFF::ESD_EXE_DATA;
      PR->NameSpace = GOFF::ESD_NS_Parts;

      GSection = createGOFFSection(PR, PR, SD, &Section);
    }
  } else
    llvm_unreachable("Unhandled section kind");

  SectionMap[&Section] = GSection;
}

void GOFFObjectWriter::processSymbolDefinedInModule(
    const MCSymbolGOFF &MCSymbol, const MCAssembler &Asm,
    const MCAsmLayout &Layout) {
  MCSection &Section = MCSymbol.getSection();
  SectionKind Kind = Section.getKind();
  auto &Sec = cast<MCSectionGOFF>(Section);

  GOFFSection *GSection = SectionMap[&Sec];
  assert(GSection && "No corresponding section found");

  GOFFSymbol *GSectionSym = GSection->Pptr;
  assert(GSectionSym &&
         "Defined symbols must exist in an initialized GSection");

  StringRef SymbolName = MCSymbol.getName();
  // If it's a text section, then create a label for it.
  if (Kind.isText()) {
    GOFFSymbol *LD = createLDSymbol(SymbolName, GSectionSym->EsdId);
    LD->BindingStrength = MCSymbol.isWeak()
                              ? GOFF::ESDBindingStrength::ESD_BST_Weak
                              : GOFF::ESDBindingStrength::ESD_BST_Strong;

    // If we don't know if it is code or data, assume it is code.
    LD->Executable = MCSymbol.getExecutable();
    if (LD->isExecUnspecified())
      LD->Executable = GOFF::ESD_EXE_CODE;

    // Determine the binding scope. Please note that the combination
    // !isExternal && isExported makes no sense.
    LD->BindingScope = MCSymbol.isExternal()
                           ? (MCSymbol.isExported()
                                  ? GOFF::ESD_BSC_ImportExport
                                  : (LD->isExecutable() ? GOFF::ESD_BSC_Library
                                                        : GOFF::ESD_BSC_Module))
                           : GOFF::ESD_BSC_Section;

    if (ADA && ADA->SectionLength > 0)
      LD->ADAEsdId = ADA->EsdId;
    else
      LD->ADAEsdId = 0;

    GSectionSym->setMaxAlignment(MCSymbol.getAlignment());

    LD->MCSym = &MCSymbol;
    SymbolMap[&MCSymbol] = LD;
  } else if (Kind.isBSS() || Kind.isData()) {
    GSectionSym = GSection->Rptr;
    GSectionSym->BindingScope =
        MCSymbol.isExternal()
            ? (MCSymbol.isExported() ? GOFF::ESD_BSC_ImportExport
                                     : GOFF::ESD_BSC_Library)
            : GOFF::ESD_BSC_Section;
    if (GSectionSym->BindingScope == GOFF::ESD_BSC_Section)
      GSection->SD->BindingScope = GOFF::ESD_BSC_Section;
    GSectionSym->setAlignment(MCSymbol.getAlignment());
    GSectionSym->SectionLength = Layout.getSectionAddressSize(&Section);

    // We cannot have a zero-length section for data.  If we do, artificially
    // inflate it.  Use 2 bytes to avoid odd alignments.
    if (!GSectionSym->SectionLength)
      GSectionSym->SectionLength = 2;

    GSectionSym->MCSym = &MCSymbol;
    GSection->SD->MCSym = &MCSymbol;
    SymbolMap[&MCSymbol] = GSectionSym;
  } else
    llvm_unreachable("Unhandled section kind for Symbol");
}

void GOFFObjectWriter::processSymbolDeclaredInModule(
    const MCSymbolGOFF &Symbol) {
  GOFFSymbol *SD = RootSD;

  switch (Symbol.getExecutable()) {
  case GOFF::ESD_EXE_CODE:
  case GOFF::ESD_EXE_Unspecified: {
    GOFFSymbol *ER = createERSymbol(Symbol.getName(), SD->EsdId, &Symbol);
    ER->BindingScope = GOFF::ESD_BSC_ImportExport;
    ER->MCSym = &Symbol;
    SymbolMap[&Symbol] = ER;
    break;
  }
  case GOFF::ESD_EXE_DATA: {
    GOFFSymbol *ED = createWSASymbol(SD->EsdId);
    ED->setAlignment(Symbol.getAlignment());
    GOFFSymbol *PR = createPRSymbol(Symbol.getName(), ED->EsdId);

    PR->BindingScope = GOFF::ESD_BSC_ImportExport;
    PR->setAlignment(Symbol.getAlignment());
    PR->NameSpace = GOFF::ESD_NS_Parts;
    PR->SectionLength = 0;
    PR->MCSym = &Symbol;
    SymbolMap[&Symbol] = PR;
    break;
  }
  }
}

void GOFFObjectWriter::executePostLayoutBinding(MCAssembler &Asm,
                                                const MCAsmLayout &Layout) {
  LLVM_DEBUG(dbgs() << "Entering " << __FUNCTION__ << "\n");

  // Define the GOFF root and ADA symbol.
  defineRootAndADASD(Asm);

  for (MCSection &S : Asm) {
    auto &Section = cast<MCSectionGOFF>(S);
    LLVM_DEBUG(dbgs() << "Current Section (" << Section.getName() << "): ";
               Section.dump(); dbgs() << "\n");
    defineSectionSymbols(Asm, Section, Layout);
  }

  for (const MCSymbol &Sym : Asm.symbols()) {
    if (Sym.isTemporary() && !Sym.isUsedInReloc())
      continue;

    auto &Symbol = cast<MCSymbolGOFF>(Sym);
    if (Symbol.isDefined())
      processSymbolDefinedInModule(Symbol, Asm, Layout);
    else
      processSymbolDeclaredInModule(Symbol);
  }
}

void GOFFObjectWriter::writeHeader() {
  OS.newRecord(GOFF::RT_HDR, /*Size=*/57);
  OS.write_zeros(1);       // Reserved
  OS.writebe<uint32_t>(0); // Target Hardware Environment
  OS.writebe<uint32_t>(0); // Target Operating System Environment
  OS.write_zeros(2);       // Reserved
  OS.writebe<uint16_t>(0); // CCSID
  OS.write_zeros(16);      // Character Set name
  OS.write_zeros(16);      // Language Product Identifier
  OS.writebe<uint32_t>(1); // Architecture Level
  OS.writebe<uint16_t>(0); // Module Properties Length
  OS.write_zeros(6);       // Reserved
}

void GOFFObjectWriter::writeSymbol(const GOFFSymbol &Symbol,
                                   const MCAsmLayout &Layout) {
  uint32_t Offset = 0;
  uint32_t Length = 0;
  GOFF::ESDNameSpaceId NameSpaceId = GOFF::ESD_NS_ProgramManagementBinder;
  Flags SymbolFlags;
  uint8_t FillByteValue = 0;

  Flags BehavAttrs[10] = {};
  auto setAmode = [&BehavAttrs](GOFF::ESDAmode Amode) {
    BehavAttrs[0].set(0, 8, Amode);
  };
  auto setRmode = [&BehavAttrs](GOFF::ESDRmode Rmode) {
    BehavAttrs[1].set(0, 8, Rmode);
  };
  auto setTextStyle = [&BehavAttrs](GOFF::ESDTextStyle Style) {
    BehavAttrs[2].set(0, 4, Style);
  };
  auto setBindingAlgorithm =
      [&BehavAttrs](GOFF::ESDBindingAlgorithm Algorithm) {
        BehavAttrs[2].set(4, 4, Algorithm);
      };
  auto setTaskingBehavior =
      [&BehavAttrs](GOFF::ESDTaskingBehavior TaskingBehavior) {
        BehavAttrs[3].set(0, 3, TaskingBehavior);
      };
  auto setReadOnly = [&BehavAttrs](bool ReadOnly) {
    BehavAttrs[3].set(4, 1, ReadOnly);
  };
  auto setExecutable = [&BehavAttrs](GOFF::ESDExecutable Executable) {
    BehavAttrs[3].set(5, 3, Executable);
  };
  auto setDuplicateSeverity =
      [&BehavAttrs](GOFF::ESDDuplicateSymbolSeverity DSS) {
        BehavAttrs[4].set(2, 2, DSS);
      };
  auto setBindingStrength = [&BehavAttrs](GOFF::ESDBindingStrength Strength) {
    BehavAttrs[4].set(4, 4, Strength);
  };
  auto setLoadingBehavior = [&BehavAttrs](GOFF::ESDLoadingBehavior Behavior) {
    BehavAttrs[5].set(0, 2, Behavior);
  };
  auto setIndirectReference = [&BehavAttrs](bool Indirect) {
    uint8_t Value = Indirect ? 1 : 0;
    BehavAttrs[5].set(3, 1, Value);
  };
  auto setBindingScope = [&BehavAttrs](GOFF::ESDBindingScope Scope) {
    BehavAttrs[5].set(4, 4, Scope);
  };
  auto setLinkageType = [&BehavAttrs](GOFF::ESDLinkageType Type) {
    BehavAttrs[6].set(2, 1, Type);
  };
  auto setAlignment = [&BehavAttrs](GOFF::ESDAlignment Alignment) {
    BehavAttrs[6].set(3, 5, Alignment);
  };

  uint32_t AdaEsdId = 0;
  uint32_t SortPriority = 0;

  switch (Symbol.SymbolType) {
  case GOFF::ESD_ST_SectionDefinition: {
    if (Symbol.isExecutable()) // Unspecified otherwise
      setTaskingBehavior(GOFF::ESD_TA_Rent);
    if (Symbol.BindingScope == GOFF::ESD_BSC_Section)
      setBindingScope(Symbol.BindingScope);
  } break;
  case GOFF::ESD_ST_ElementDefinition: {
    SymbolFlags.set(3, 1, Symbol.isRemovable()); // Removable
    if (Symbol.isExecutable()) {
      setExecutable(GOFF::ESD_EXE_CODE);
      setReadOnly(true);
    } else {
      if (Symbol.isExecUnspecified())
        setExecutable(GOFF::ESD_EXE_Unspecified);
      else
        setExecutable(GOFF::ESD_EXE_DATA);

      if (Symbol.isForceRent() || Symbol.isReadOnly()) // TODO
        setReadOnly(true);
    }
    Offset = 0; // TODO ED and SD are 1-1 for now
    setAlignment(Symbol.Alignment);
    SymbolFlags.set(0, 1, 1); // Fill-Byte Value Presence Flag
    FillByteValue = 0;
    SymbolFlags.set(1, 1, 0); // Mangled Flag TODO ?
    setAmode(Symbol.Amode);
    setRmode(Symbol.Rmode);
    setTextStyle(Symbol.TextStyle);
    setBindingAlgorithm(Symbol.BindAlgorithm);
    setLoadingBehavior(Symbol.LoadBehavior);
    SymbolFlags.set(5, 3, GOFF::ESD_RQ_0); // Reserved Qwords
    if (Symbol.isForceRent())
      setReadOnly(true);
    NameSpaceId = Symbol.NameSpace;
    Length = Symbol.SectionLength;
    break;
  }
  case GOFF::ESD_ST_LabelDefinition: {
    if (Symbol.isExecutable())
      setExecutable(GOFF::ESD_EXE_CODE);
    else
      setExecutable(GOFF::ESD_EXE_DATA);
    setBindingStrength(Symbol.BindingStrength);
    setLinkageType(Symbol.Linkage);
    SymbolFlags.set(2, 1, Symbol.Renamable); // Renamable;
    setAmode(Symbol.Amode);
    NameSpaceId = Symbol.NameSpace;
    setBindingScope(Symbol.BindingScope);
    AdaEsdId = Symbol.ADAEsdId;

    // Only symbol that doesn't have an MC is the SectionLabelSymbol which
    // implicitly has 0 offset into the parent SD!
    if (auto *MCSym = Symbol.MCSym) {
      uint64_t Ofs = Layout.getSymbolOffset(*MCSym);
      // We only have signed 32bits of offset!
      assert(Ofs < (((uint64_t)1) << 31) && "ESD offset out of range.");
      Offset = static_cast<uint32_t>(Ofs);
    }
    break;
  }
  case GOFF::ESD_ST_ExternalReference: {
    setExecutable(Symbol.isExecutable() ? GOFF::ESD_EXE_CODE
                                        : GOFF::ESD_EXE_DATA);
    setBindingStrength(Symbol.BindingStrength);
    setLinkageType(Symbol.Linkage);
    SymbolFlags.set(2, 1, Symbol.Renamable); // Renamable;
    setIndirectReference(Symbol.Indirect);
    Offset = 0; // ERs don't do offsets
    NameSpaceId = Symbol.NameSpace;
    setBindingScope(Symbol.BindingScope);
    setAmode(Symbol.Amode);
    break;
  }
  case GOFF::ESD_ST_PartReference: {
    setExecutable(Symbol.isExecutable() ? GOFF::ESD_EXE_CODE
                                        : GOFF::ESD_EXE_DATA);
    NameSpaceId = Symbol.NameSpace;
    setAlignment(Symbol.Alignment);
    setAmode(Symbol.Amode);
    setLinkageType(Symbol.Linkage);
    setBindingScope(Symbol.BindingScope);
    SymbolFlags.set(2, 1, Symbol.Renamable); // Renamable;
    setDuplicateSeverity(Symbol.isWeakRef() ? GOFF::ESD_DSS_NoWarning
                                            : GOFF::ESD_DSS_Warning);
    setIndirectReference(Symbol.Indirect);
    setReadOnly(Symbol.ReadOnly);
    SortPriority = Symbol.SortKey;

    Length = Symbol.SectionLength;
    break;
  }
  } // End switch

  SmallString<256> Res;
  ConverterEBCDIC::convertToEBCDIC(Symbol.Name, Res);
  StringRef Name = Res.str();

  // Assert here since this number is technically signed but we need uint for
  // writing to records.
  assert(Name.size() < GOFF::MaxDataLength &&
         "Symbol max name length exceeded");
  uint16_t NameLength = Name.size();

  OS.newRecord(GOFF::RT_ESD, GOFF::ESDMetadataLength + NameLength);
  OS.writebe<uint8_t>(Symbol.SymbolType);       // Symbol Type
  OS.writebe<uint32_t>(Symbol.EsdId);           // ESDID
  OS.writebe<uint32_t>(Symbol.ParentEsdId);     // Parent or Owning ESDID
  OS.writebe<uint32_t>(0);                      // Reserved
  OS.writebe<uint32_t>(Offset);                 // Offset or Address
  OS.writebe<uint32_t>(0);                      // Reserved
  OS.writebe<uint32_t>(Length);                 // Length
  OS.writebe<uint32_t>(Symbol.EASectionEsdId);  // Extended Attribute ESDID
  OS.writebe<uint32_t>(Symbol.EASectionOffset); // Extended Attribute Offset
  OS.writebe<uint32_t>(0);                      // Reserved
  OS.writebe<uint8_t>(NameSpaceId);             // Name Space ID
  OS.writebe<uint8_t>(SymbolFlags);             // Flags
  OS.writebe<uint8_t>(FillByteValue);           // Fill-Byte Value
  OS.writebe<uint8_t>(0);                       // Reserved
  OS.writebe<uint32_t>(AdaEsdId);               // ADA ESDID
  OS.writebe<uint32_t>(SortPriority);           // Sort Priority
  OS.writebe<uint64_t>(0);                      // Reserved
  for (auto F : BehavAttrs)
    OS.writebe<uint8_t>(F);          // Behavioral Attributes
  OS.writebe<uint16_t>(NameLength);  // Name Length
  OS.write(Name.data(), NameLength); // Name
}

namespace {
/// Adapter stream to write a text section.
class TextStream : public raw_ostream {
  /// The underlying GOFFOstream.
  GOFFOstream &OS;

  /// The buffer size is the maximum number of bytes in a TXT section.
  static constexpr size_t BufferSize = GOFF::MaxDataLength;

  /// Static allocated buffer for the stream, used by the raw_ostream class. The
  /// buffer is sized to hold the payload of a logical TXT record.
  char Buffer[BufferSize];

  /// The offset for the next TXT record. This is equal to the number of bytes
  /// written.
  size_t Offset;

  /// The Esdid of the GOFF section.
  const uint32_t EsdId;

  /// The record style.
  const GOFF::TXTRecordStyle RecordStyle;

  /// See raw_ostream::write_impl.
  void write_impl(const char *Ptr, size_t Size) override;

  uint64_t current_pos() const override { return Offset; }

public:
  explicit TextStream(GOFFOstream &OS, uint32_t EsdId,
                      GOFF::TXTRecordStyle RecordStyle)
      : OS(OS), Offset(0), EsdId(EsdId), RecordStyle(RecordStyle) {
    SetBuffer(Buffer, sizeof(Buffer));
  }

  ~TextStream() { flush(); }
};

void TextStream::write_impl(const char *Ptr, size_t Size) {
  size_t WrittenLength = 0;

  // We only have signed 32bits of offset.
  if (Offset + Size > std::numeric_limits<int32_t>::max())
    report_fatal_error("TXT section too large");

  while (WrittenLength < Size) {
    size_t ToWriteLength =
        std::min(Size - WrittenLength, size_t(GOFF::MaxDataLength));

    OS.newRecord(GOFF::RT_TXT, GOFF::TXTMetadataLength + ToWriteLength);
    OS.writebe<uint8_t>(Flags(4, 4, RecordStyle));       // Text Record Style
    OS.writebe<uint32_t>(EsdId);                         // Element ESDID
    OS.writebe<uint32_t>(0);                             // Reserved
    OS.writebe<uint32_t>(static_cast<uint32_t>(Offset)); // Offset
    OS.writebe<uint32_t>(0);                      // Text Field True Length
    OS.writebe<uint16_t>(0);                      // Text Encoding
    OS.writebe<uint16_t>(ToWriteLength);          // Data Length
    OS.write(Ptr + WrittenLength, ToWriteLength); // Data

    WrittenLength += ToWriteLength;
    Offset += ToWriteLength;
  }
}
} // namespace

void GOFFObjectWriter::writeText(const GOFFSection &Section,
                                 const MCAssembler &Asm,
                                 const MCAsmLayout &Layout) {
  // TODO: This assumes the ED. Is that correct?  Probably not.
  TextStream S(OS, Section.Pptr->EsdId,
               Section.IsStructured ? GOFF::TXT_RS_Structured
                                    : GOFF::TXT_RS_Byte);
  Asm.writeSectionData(S, Section.MCSec, Layout);
}

void GOFFObjectWriter::writeEnd() {
  uint8_t F = GOFF::END_EPR_None;
  uint8_t AMODE = 0;
  uint32_t ESDID = 0;

  // TODO Set Flags/AMODE/ESDID for entry point.

  OS.newRecord(GOFF::RT_END, /*Size=*/13);
  OS.writebe<uint8_t>(Flags(6, 2, F)); // Indicator flags
  OS.writebe<uint8_t>(AMODE);          // AMODE
  OS.write_zeros(3);                   // Reserved
  // The record count is the number of logical records. In principle, this value
  // is available as OS.logicalRecords(). However, some tools rely on this field
  // being zero.
  OS.writebe<uint32_t>(0);     // Record Count
  OS.writebe<uint32_t>(ESDID); // ESDID (of entry point)
  OS.finalize();
}

uint64_t GOFFObjectWriter::writeObject(MCAssembler &Asm,
                                       const MCAsmLayout &Layout) {
  uint64_t StartOffset = OS.tell();

  writeHeader();
  for (const auto &Sym : EsdSymbols)
    writeSymbol(*Sym, Layout);
  for (const auto &Sec : Sections)
    writeText(*Sec, Asm, Layout);
  writeEnd();

  LLVM_DEBUG(dbgs() << "Wrote " << OS.logicalRecords() << " logical records.");

  return OS.tell() - StartOffset;
}

std::unique_ptr<MCObjectWriter>
llvm::createGOFFObjectWriter(std::unique_ptr<MCGOFFObjectTargetWriter> MOTW,
                             raw_pwrite_stream &OS) {
  return std::make_unique<GOFFObjectWriter>(std::move(MOTW), OS);
}
