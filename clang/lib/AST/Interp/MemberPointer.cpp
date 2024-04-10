

#include "MemberPointer.h"
#include "Context.h"
#include "FunctionPointer.h"
#include "Program.h"
#include "Record.h"

namespace clang {
namespace interp {

std::optional<Pointer> MemberPointer::toPointer(const Context &Ctx) const {
  if (!Dcl || isa<FunctionDecl>(Dcl))
    return Base;
  const FieldDecl *FD = cast<FieldDecl>(Dcl);
  assert(FD);

  Pointer CastedBase =
      (PtrOffset < 0 ? Base.atField(-PtrOffset) : Base.atFieldSub(PtrOffset));

  const Record *BaseRecord = CastedBase.getRecord();
  if (!BaseRecord)
    return std::nullopt;

  assert(BaseRecord);
  if (FD->getParent() == BaseRecord->getDecl())
    return CastedBase.atField(BaseRecord->getField(FD)->Offset);

  unsigned Offset;
  const RecordDecl *FieldParent = FD->getParent();
  const Record *FieldRecord = Ctx.getRecord(FieldParent);

  Offset = 0;
  Offset += FieldRecord->getField(FD)->Offset;
  Offset += CastedBase.block()->getDescriptor()->getMetadataSize();

  if (Offset > CastedBase.block()->getSize())
    return std::nullopt;

  unsigned O = 0;
  if (Base.getDeclPtr().getRecord()->getDecl() != FieldParent) {
    O = Ctx.collectBaseOffset(FieldParent,
                              Base.getDeclPtr().getRecord()->getDecl());
    Offset += O;
  }

  if (Offset > CastedBase.block()->getSize())
    return std::nullopt;

  assert(Offset <= CastedBase.block()->getSize());
  Pointer Result;
  Result = Pointer(const_cast<Block *>(Base.block()), Offset, Offset);
  return Result;
}

FunctionPointer MemberPointer::toFunctionPointer(const Context &Ctx) const {
  return FunctionPointer(Ctx.getProgram().getFunction(cast<FunctionDecl>(Dcl)));
}

APValue MemberPointer::toAPValue() const {
  if (isZero())
    return APValue(static_cast<ValueDecl *>(nullptr), /*IsDerivedMember=*/false,
                   /*Path=*/{});

  if (hasBase())
    return Base.toAPValue();

  return APValue(cast<ValueDecl>(getDecl()), /*IsDerivedMember=*/false,
                 /*Path=*/{});
}

} // namespace interp
} // namespace clang
