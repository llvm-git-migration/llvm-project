#include "Basic/SequenceToOffsetTable.h"
#include "Common/CodeGenDAGPatterns.h" // For SDNodeInfo.
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/StringToOffsetTable.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

static cl::OptionCategory SDNodeInfoEmitterCat("Options for -gen-sdnode-info");

static cl::opt<std::string> TargetSDNodeNamespace(
    "sdnode-namespace", cl::cat(SDNodeInfoEmitterCat),
    cl::desc("Specify target SDNode namespace (default=<Target>ISD)"));

namespace {

class SDNodeInfoEmitter {
  const RecordKeeper &RK;
  const CodeGenTarget Target;
  std::vector<SDNodeInfo> AllNodes;
  std::map<StringRef, SmallVector<const SDNodeInfo *, 2>> TargetNodesByName;

public:
  explicit SDNodeInfoEmitter(const RecordKeeper &RK);

  void run(raw_ostream &OS) const;

private:
  void emitNodeEnum(raw_ostream &OS) const;
  void emitNodeNames(raw_ostream &OS) const;
  void emitTypeConstraints(raw_ostream &OS) const;
  void emitNodeDescs(raw_ostream &OS) const;
};

} // namespace

SDNodeInfoEmitter::SDNodeInfoEmitter(const RecordKeeper &RK)
    : RK(RK), Target(RK) {
  const CodeGenHwModes &HwModes = Target.getHwModes();

  if (!TargetSDNodeNamespace.getNumOccurrences())
    TargetSDNodeNamespace = Target.getName().str() + "ISD";

  for (const Record *R : RK.getAllDerivedDefinitions("SDNode"))
    AllNodes.emplace_back(R, HwModes);

  for (const SDNodeInfo &Node : AllNodes) {
    StringRef QualifiedName = Node.getEnumName();
    auto [NS, Name] = QualifiedName.split("::");

    if (NS == TargetSDNodeNamespace)
      TargetNodesByName[Name].push_back(&Node);
  }
}

void SDNodeInfoEmitter::emitNodeEnum(raw_ostream &OS) const {
  OS << "#ifdef GET_SDNODE_ENUM\n";
  OS << "#undef GET_SDNODE_ENUM\n\n";
  OS << "namespace llvm::" << TargetSDNodeNamespace << " {\n\n";

  OS << "enum GenNodeType : unsigned {\n";

  if (!TargetNodesByName.empty()) {
    StringRef FirstName = TargetNodesByName.begin()->first;
    OS << "  " << FirstName << " = ISD::BUILTIN_OP_END,\n";
    for (StringRef Name : make_first_range(drop_begin(TargetNodesByName)))
      OS << "  " << Name << ",\n";
  }

  OS << "};\n\n";

  if (!TargetNodesByName.empty()) {
    StringRef LastName = TargetNodesByName.rbegin()->first;
    OS << "static constexpr unsigned GENERATED_OPCODE_END = " << LastName
       << " + 1;\n\n";
  }

  OS << "} // namespace llvm::" << TargetSDNodeNamespace << "\n\n";
  OS << "#endif // GET_SDNODE_ENUM\n\n";
}

void SDNodeInfoEmitter::emitNodeNames(raw_ostream &OS) const {
  StringRef TargetName = Target.getName();
  StringToOffsetTable NodeNameTable;

  OS << "static const unsigned " << TargetName << "NodeNameOffsets["
     << TargetNodesByName.size() << "] = {";

  std::string DebugName;
  for (auto [Idx, Name] : enumerate(make_first_range(TargetNodesByName))) {
    // Newline every 8 entries.
    OS << (Idx % 8 == 0 ? "\n    " : " ");
    DebugName = (TargetSDNodeNamespace + "::" + Name).str();
    OS << NodeNameTable.GetOrAddStringOffset(DebugName) << ",";
  }

  OS << "\n};\n";

  NodeNameTable.EmitStringLiteralDef(
      OS, "static const char " + TargetName + "NodeNames[]", /*Indent=*/"");

  OS << "\n";
}

static void emitConstraint(raw_ostream &OS, SDTypeConstraint C) {
  StringRef Name;
  unsigned OtherOpNo = 0;
  MVT VT;

  switch (C.ConstraintType) {
  case SDTypeConstraint::SDTCisVT:
    Name = "SDTCisVT";
    if (C.VVT.isSimple())
      VT = C.VVT.getSimple();
    break;
  case SDTypeConstraint::SDTCisPtrTy:
    Name = "SDTCisPtrTy";
    break;
  case SDTypeConstraint::SDTCisInt:
    Name = "SDTCisInt";
    break;
  case SDTypeConstraint::SDTCisFP:
    Name = "SDTCisFP";
    break;
  case SDTypeConstraint::SDTCisVec:
    Name = "SDTCisVec";
    break;
  case SDTypeConstraint::SDTCisSameAs:
    Name = "SDTCisSameAs";
    OtherOpNo = C.x.SDTCisSameAs_Info.OtherOperandNum;
    break;
  case SDTypeConstraint::SDTCisVTSmallerThanOp:
    Name = "SDTCisVTSmallerThanOp";
    OtherOpNo = C.x.SDTCisVTSmallerThanOp_Info.OtherOperandNum;
    break;
  case SDTypeConstraint::SDTCisOpSmallerThanOp:
    Name = "SDTCisOpSmallerThanOp";
    OtherOpNo = C.x.SDTCisOpSmallerThanOp_Info.BigOperandNum;
    break;
  case SDTypeConstraint::SDTCisEltOfVec:
    Name = "SDTCisEltOfVec";
    OtherOpNo = C.x.SDTCisEltOfVec_Info.OtherOperandNum;
    break;
  case SDTypeConstraint::SDTCisSubVecOfVec:
    Name = "SDTCisSubVecOfVec";
    OtherOpNo = C.x.SDTCisSubVecOfVec_Info.OtherOperandNum;
    break;
  case SDTypeConstraint::SDTCVecEltisVT:
    Name = "SDTCVecEltisVT";
    if (C.VVT.isSimple())
      VT = C.VVT.getSimple();
    break;
  case SDTypeConstraint::SDTCisSameNumEltsAs:
    Name = "SDTCisSameNumEltsAs";
    OtherOpNo = C.x.SDTCisSameNumEltsAs_Info.OtherOperandNum;
    break;
  case SDTypeConstraint::SDTCisSameSizeAs:
    Name = "SDTCisSameSizeAs";
    OtherOpNo = C.x.SDTCisSameSizeAs_Info.OtherOperandNum;
    break;
  }

  StringRef VTName = VT.SimpleTy == MVT::INVALID_SIMPLE_VALUE_TYPE
                         ? "MVT::INVALID_SIMPLE_VALUE_TYPE"
                         : getEnumName(VT.SimpleTy);
  OS << '{' << C.OperandNo << ", " << Name << ", " << OtherOpNo << ", "
     << VTName << '}';
}

void SDNodeInfoEmitter::emitTypeConstraints(raw_ostream &OS) const {
  SequenceToOffsetTable<SmallVector<SDTypeConstraint, 0>> ConstraintTable(
      /*Terminator=*/std::nullopt);
  SmallVector<StringRef> SkippedNodes;

  for (const auto &[Name, Nodes] : TargetNodesByName) {
    const SDNodeInfo *N = Nodes.front();
    ArrayRef<SDTypeConstraint> Constraints = N->getTypeConstraints();

    bool IsAmbiguous = any_of(drop_begin(Nodes), [&](const SDNodeInfo *Other) {
      return ArrayRef(Other->getTypeConstraints()) != Constraints;
    });

    if (IsAmbiguous) {
      SkippedNodes.push_back(Name);
      continue;
    }

    // Reversing the order increases the likelihood of reusing storage.
    SmallVector<SDTypeConstraint, 0> RevConstraints(reverse(Constraints));
    ConstraintTable.add(RevConstraints);
  }

  ConstraintTable.layout();

  OS << "static const SDTypeConstraint " << Target.getName()
     << "SDTypeConstraints[" << ConstraintTable.size() << "] = {\n";
  ConstraintTable.emit(OS, emitConstraint);
  OS << "};\n\n";

  unsigned NumOpcodes = TargetNodesByName.size();
  OS << "static const std::pair<unsigned, unsigned> " << Target.getName()
     << "SDTypeConstraintOffsets[" << NumOpcodes << "] = {";

  unsigned Idx = 0;
  for (const auto &[Name, Nodes] : TargetNodesByName) {
    // Newline every 8 entries.
    OS << (Idx++ % 8 == 0 ? "\n    " : " ");

    if (is_contained(SkippedNodes, Name)) {
      OS << "{0, 0},";
      continue;
    }

    ArrayRef<SDTypeConstraint> Constraints = Nodes[0]->getTypeConstraints();
    SmallVector<SDTypeConstraint, 0> RevConstraints(reverse(Constraints));
    OS << '{' << ConstraintTable.get(RevConstraints) << ", "
       << Constraints.size() << "},";
  }

  OS << "};\n\n";
}

static void emitDesc(raw_ostream &OS, StringRef Name,
                     ArrayRef<const SDNodeInfo *> Nodes) {
  const SDNodeInfo *N = Nodes.front();

  // We're only interested in a subset of node properties. Properties like
  // SDNPAssociative and SDNPCommutative do not impose constraints on nodes,
  // and sometimes differ between nodes using the same enum name.
  constexpr unsigned InterestingProperties =
      (1 << SDNPHasChain) | (1 << SDNPOutGlue) | (1 << SDNPInGlue) |
      (1 << SDNPOptInGlue) | (1 << SDNPMemOperand) | (1 << SDNPVariadic);

  unsigned NumResults = N->getNumResults();
  int NumOperands = N->getNumOperands();
  unsigned Properties = N->getProperties();
  bool IsStrictFP = N->isStrictFP();
  uint64_t TSFlags = N->getTSFlags();

  assert(all_of(drop_begin(Nodes), [&](const SDNodeInfo *Other) {
    return Other->getNumResults() == NumResults &&
           Other->getNumOperands() == NumOperands &&
           (Other->getProperties() & InterestingProperties) ==
               (Properties & InterestingProperties) &&
           Other->isStrictFP() == IsStrictFP && Other->getTSFlags() == TSFlags;
  }));

  OS << "    {" << NumResults;
  OS << ", " << NumOperands;

  OS << ", 0";
  if (Properties & (1 << SDNPHasChain))
    OS << "|1<<SDNPHasChain";
  if (Properties & (1 << SDNPOutGlue))
    OS << "|1<<SDNPOutGlue";
  if (Properties & (1 << SDNPInGlue))
    OS << "|1<<SDNPInGlue";
  if (Properties & (1 << SDNPOptInGlue))
    OS << "|1<<SDNPOptInGlue";
  if (Properties & (1 << SDNPVariadic))
    OS << "|1<<SDNPVariadic";
  if (Properties & (1 << SDNPMemOperand))
    OS << "|1<<SDNPMemOperand";

  OS << ", 0";
  if (IsStrictFP)
    OS << "|1<<SDNFIsStrictFP";

  OS << ", " << TSFlags;
  OS << "}," << " // " << Name << '\n';
}

void SDNodeInfoEmitter::emitNodeDescs(raw_ostream &OS) const {
  StringRef TargetName = Target.getName();

  OS << "#ifdef GET_SDNODE_DESC\n";
  OS << "#undef GET_SDNODE_DESC\n\n";
  OS << "namespace llvm {\n\n";

  emitNodeNames(OS);
  emitTypeConstraints(OS);

  unsigned NumOpcodes = TargetNodesByName.size();
  OS << "static const SDNodeDesc " << TargetName << "NodeDescs[" << NumOpcodes
     << "] = {\n";

  for (const auto &[Name, Nodes] : TargetNodesByName)
    emitDesc(OS, Name, Nodes);

  OS << "};\n\n";

  OS << "static const SDNodeInfo " << TargetName << "GenSDNodeInfo(\n    "
     << NumOpcodes << ", " << TargetName << "NodeDescs,\n    " << TargetName
     << "NodeNames, " << TargetName << "NodeNameOffsets,\n    " << TargetName
     << "SDTypeConstraints, " << TargetName << "SDTypeConstraintOffsets);\n\n";

  OS << "} // namespace llvm\n\n";
  OS << "#endif // GET_SDNODE_DESC\n\n";
}

void SDNodeInfoEmitter::run(raw_ostream &OS) const {
  emitSourceFileHeader("Target SDNode descriptions", OS, RK);
  emitNodeEnum(OS);
  emitNodeDescs(OS);
}

static TableGen::Emitter::OptClass<SDNodeInfoEmitter>
    X("gen-sd-node-info", "Generate target SDNode descriptions");
