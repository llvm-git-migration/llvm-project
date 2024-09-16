//===-- DWARFIndex.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DWARFIndex.h"
#include "DWARFDebugInfoEntry.h"
#include "DWARFDeclContext.h"
#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"

#include "lldb/Core/Mangled.h"
#include "lldb/Core/Module.h"
#include "lldb/Target/Language.h"

using namespace lldb_private;
using namespace lldb;
using namespace lldb_private::plugin::dwarf;

DWARFIndex::~DWARFIndex() = default;

bool DWARFIndex::ProcessFunctionDIE(
    const Module::LookupInfo &lookup_info, DWARFDIE die,
    const CompilerDeclContext &parent_decl_ctx,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  llvm::StringRef name = lookup_info.GetLookupName().GetStringRef();
  FunctionNameType name_type_mask = lookup_info.GetNameTypeMask();

  if (!(name_type_mask & eFunctionNameTypeFull)) {
    ConstString name_to_match_against;
    if (const char *mangled_die_name = die.GetMangledName()) {
      name_to_match_against = ConstString(mangled_die_name);
    } else {
      SymbolFileDWARF *symbols = die.GetDWARF();
      if (ConstString demangled_die_name =
              symbols->ConstructFunctionDemangledName(die))
        name_to_match_against = demangled_die_name;
    }

    if (!lookup_info.NameMatchesLookupInfo(name_to_match_against,
                                           lookup_info.GetLanguageType()))
      return true;
  }

  // Exit early if we're searching exclusively for methods or selectors and
  // we have a context specified (no methods in namespaces).
  uint32_t looking_for_nonmethods =
      name_type_mask & ~(eFunctionNameTypeMethod | eFunctionNameTypeSelector);
  if (!looking_for_nonmethods && parent_decl_ctx.IsValid())
    return true;

  // Otherwise, we need to also check that the context matches. If it does not
  // match, we do nothing.
  if (!SymbolFileDWARF::DIEInDeclContext(parent_decl_ctx, die))
    return true;

  // In case of a full match, we just insert everything we find.
  if (name_type_mask & eFunctionNameTypeFull && die.GetMangledName() == name)
    return callback(die);

  // If looking for ObjC selectors, we need to also check if the name is a
  // possible selector.
  if (name_type_mask & eFunctionNameTypeSelector &&
      ObjCLanguage::IsPossibleObjCMethodName(die.GetName()))
    return callback(die);

  bool looking_for_methods = name_type_mask & lldb::eFunctionNameTypeMethod;
  bool looking_for_functions = name_type_mask & lldb::eFunctionNameTypeBase;
  if (looking_for_methods || looking_for_functions) {
    // If we're looking for either methods or functions, we definitely want this
    // die. Otherwise, only keep it if the die type matches what we are
    // searching for.
    if ((looking_for_methods && looking_for_functions) ||
        looking_for_methods == die.IsMethod())
      return callback(die);
  }

  return true;
}

DWARFIndex::DIERefCallbackImpl::DIERefCallbackImpl(
    const DWARFIndex &index, llvm::function_ref<bool(DWARFDIE die)> callback,
    llvm::StringRef name)
    : m_index(index),
      m_dwarf(*llvm::cast<SymbolFileDWARF>(
          index.m_module.GetSymbolFile()->GetBackingSymbolFile())),
      m_callback(callback), m_name(name) {}

bool DWARFIndex::DIERefCallbackImpl::operator()(DIERef ref) const {
  if (DWARFDIE die = m_dwarf.GetDIE(ref))
    return m_callback(die);
  m_index.ReportInvalidDIERef(ref, m_name);
  return true;
}

bool DWARFIndex::DIERefCallbackImpl::operator()(
    const llvm::AppleAcceleratorTable::Entry &entry) const {
  return this->operator()(DIERef(std::nullopt, DIERef::Section::DebugInfo,
                                 *entry.getDIESectionOffset()));
}

void DWARFIndex::ReportInvalidDIERef(DIERef ref, llvm::StringRef name) const {
  m_module.ReportErrorIfModifyDetected(
      "the DWARF debug information has been modified (accelerator table had "
      "bad die {0:x16} for '{1}')\n",
      ref.die_offset(), name.str().c_str());
}

void DWARFIndex::GetFullyQualifiedType(
    const DWARFDeclContext &context,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  GetTypes(context, [&](DWARFDIE die) {
    return GetFullyQualifiedTypeImpl(context, die, callback);
  });
}

bool DWARFIndex::GetFullyQualifiedTypeImpl(
    const DWARFDeclContext &context, DWARFDIE die,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  DWARFDeclContext dwarf_decl_ctx = die.GetDWARFDeclContext();
  if (dwarf_decl_ctx == context)
    return callback(die);
  return true;
}

void DWARFIndex::GetNamespacesWithParents(
    ConstString name, llvm::ArrayRef<llvm::StringRef> parent_names,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  GetNamespaces(name, [&](DWARFDIE die) {
    return ProcessDieMatchParentNames(name, parent_names, die, callback);
  });
}

void DWARFIndex::GetTypesWithParents(
    ConstString name, llvm::ArrayRef<llvm::StringRef> parent_names,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  GetTypes(name, [&](DWARFDIE die) {
    return ProcessDieMatchParentNames(name, parent_names, die, callback);
  });
}

bool DWARFIndex::ProcessDieMatchParentNames(
    ConstString name, llvm::ArrayRef<llvm::StringRef> query_parent_names,
    DWARFDIE die, llvm::function_ref<bool(DWARFDIE die)> callback) {
  std::vector<lldb_private::CompilerContext> type_context =
      die.GetTypeLookupContext();
  if (type_context.empty()) {
    // If both type_context and query_parent_names and empty we have a match.
    // Otherwise, this one does not match and we keep on searching. 
    if (query_parent_names.empty())
      return callback(die);
    return true;
  }

  // Type lookup context includes the current DIE as the last element.
  // so revert it for easy matching.
  std::reverse(type_context.begin(), type_context.end());

  // type_context includes the name of the current DIE while query_parent_names
  // doesn't. So start check from index 1 for dwarf_decl_ctx.
  uint32_t i = 1, j = 0;
  while (i < type_context.size() && j < query_parent_names.size()) {
    // If type_context[i] has no name, skip it.
    // e.g. this can happen for anonymous namespaces.
    if (type_context[i].name.IsNull() || type_context[i].name.IsEmpty()) {
      ++i;
      continue;
    }
    // If the name doesn't match, skip it.
    // e.g. this can happen for inline namespaces.
    if (query_parent_names[j] != type_context[i].name) {
      ++i;
      continue;
    }
    ++i;
    ++j;
  }
  // If not all query_parent_names were found in type_context.
  // This die does not meet the criteria, try next one.
  if (j != query_parent_names.size())
    return true;
  return callback(die);
}
