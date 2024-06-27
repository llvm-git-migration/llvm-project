// RUN: llvm-tblgen -gen-searchable-tables -I %p/../../include %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR -gen-searchable-tables -I %p/../../include %s 2>&1 \
// RUN:         | FileCheck  --check-prefix=ERROR %s

include "llvm/TableGen/SearchableTable.td"

class SysReg<string name, bits<12> op> {
  string Name = name;
  bits<12> Encoding = op;
  code FeaturesRequired = [{ {} }];
}

def List1 : GenericTable {
  let FilterClass = "SysReg";
  let Fields = [
     "Name", "Encoding", "FeaturesRequired",
  ];

  let PrimaryKey = [ "Encoding" ];
  let PrimaryKeyName = "lookupSysRegByEncoding";
  let PrimaryKeyReturnRange = true;
}

let FeaturesRequired = [{ {Feature1} }] in {
def : SysReg<"csr1", 0x7C0>;
}

let FeaturesRequired = [{ {Feature2} }] in {
def : SysReg<"csr2", 0x7C0>;
}

def lookupSysRegByName : SearchIndex {
  let Table = List1;
  let Key = [ "Name" ];
#ifdef ERROR
  // ERROR: Emitting different signature for returning a range of results is only supported for Primary Key.
  let ReturnRange = true;
#endif
}

// CHECK: #ifdef GET_List1_DECL
// CHECK-NEXT: llvm::iterator_range<const SysReg *> lookupSysRegByEncoding(uint16_t Encoding);
// CHECK-NEXT: onst SysReg *lookupSysRegByName(StringRef Name);
// CHECK-NEXT: #endif

// CHECK: #ifdef GET_List1_IMPL
// CHECK-NEXT: constexpr SysReg List1[] = {
// CHECK-NEXT:   { "csr1", 0x7C0,  {Feature1}  }, // 0
// CHECK-NEXT:   { "csr2", 0x7C0,  {Feature2}  }, // 1
// CHECK-NEXT:  };

// CHECK: llvm::iterator_range<const SysReg *> lookupSysRegByEncoding(uint16_t Encoding) {
// CHECK-NEXT: struct KeyType {
// CHECK-NEXT:    uint16_t Encoding;
// CHECK-NEXT:  };
// CHECK-NEXT:  KeyType Key = {Encoding};
// CHECK-NEXT:  struct Comp {
// CHECK-NEXT:    bool operator()(const SysReg &LHS, const KeyType &RHS) const {
// CHECK-NEXT:      if (LHS.Encoding < RHS.Encoding)
// CHECK-NEXT:        return true;
// CHECK-NEXT:      if (LHS.Encoding > RHS.Encoding)
// CHECK-NEXT:        return false;
// CHECK-NEXT:      return false;
// CHECK-NEXT:    }
// CHECK-NEXT:    bool operator()(const KeyType &LHS, const SysReg &RHS) const {
// CHECK-NEXT:      if (LHS.Encoding < RHS.Encoding)
// CHECK-NEXT:        return true;
// CHECK-NEXT:      if (LHS.Encoding > RHS.Encoding)
// CHECK-NEXT:        return false;
// CHECK-NEXT:      return false;
// CHECK-NEXT:    }
// CHECK-NEXT:  };
// CHECK-NEXT:  auto Table = ArrayRef(List1);
// CHECK-NEXT:  auto It = std::equal_range(Table.begin(), Table.end(), Key, Comp());
// CHECK-NEXT:  return llvm::make_range(It.first, It.second);
// CHECK-NEXT: }

// CHECK: const SysReg *lookupSysRegByName(StringRef Name) {
// CHECK-NEXT:   struct IndexType {
// CHECK-NEXT:     const char * Name;
// CHECK-NEXT:     unsigned _index;
// CHECK-NEXT:   };
// CHECK-NEXT:   static const struct IndexType Index[] = {
// CHECK-NEXT:     { "CSR1", 0 },
// CHECK-NEXT:     { "CSR2", 1 },
// CHECK-NEXT:   };

// CHECK:   struct KeyType {
// CHECK-NEXT:     std::string Name;
// CHECK-NEXT:   };
// CHECK-NEXT:   KeyType Key = {Name.upper()};
// CHECK-NEXT:   struct Comp {
// CHECK-NEXT:     bool operator()(const IndexType &LHS, const KeyType &RHS) const {
// CHECK-NEXT:       int CmpName = StringRef(LHS.Name).compare(RHS.Name);
// CHECK-NEXT:       if (CmpName < 0) return true;
// CHECK-NEXT:       if (CmpName > 0) return false;
// CHECK-NEXT:       return false;
// CHECK-NEXT:     }
// CHECK-NEXT:   };
// CHECK-NEXT:   auto Table = ArrayRef(Index);
// CHECK-NEXT:   auto Idx = std::lower_bound(Table.begin(), Table.end(), Key, Comp());
// CHECK-NEXT:   if (Idx == Table.end() ||
// CHECK-NEXT:       Key.Name != Idx->Name)
// CHECK-NEXT:     return nullptr;

// CHECK:  return &List1[Idx->_index];
// CHECK-NEXT: }
// CHECK-NEXT: #endif
