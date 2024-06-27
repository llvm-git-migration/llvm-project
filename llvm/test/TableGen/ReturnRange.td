// RUN: llvm-tblgen -gen-searchable-tables -I %p/../../include %s | FileCheck %s

include "llvm/TableGen/SearchableTable.td"

class SysReg<string name, bits<12> op> {
  string Name = name;
  bits<12> Encoding = op;
  code FeaturesRequired = [{ {} }];
}

def List : GenericTable {
  let FilterClass = "SysReg";
  let Fields = [
     "Name", "Encoding", "FeaturesRequired",
  ];

  let PrimaryKey = [ "Encoding" ];
  let PrimaryKeyName = "lookupSysRegByEncoding";
  let PrimaryKeyReturnRange = true;
}

def lookupSysRegByName : SearchIndex {
  let Table = List;
  let Key = [ "Name" ];
  let ReturnRange = true;
}

let FeaturesRequired = [{ {Feature1} }] in {
def : SysReg<"csr1", 0x7C0>;
}

let FeaturesRequired = [{ {Feature2} }] in {
def : SysReg<"csr2", 0x7C0>;
}

let FeaturesRequired = [{ {Feature1} }] in {
def : SysReg<"aabbc", 0x7C1>;
}

let FeaturesRequired = [{ {Feature2} }] in {
def : SysReg<"aaabb", 0x7C2>;
}

// CHECK: #ifdef GET_List_DECL
// CHECK-NEXT: llvm::iterator_range<const SysReg *>
// CHECK-NEXT: lookupSysRegByEncoding(uint16_t Encoding);
// CHECK-NEXT: llvm::iterator_range<const SysReg *>
// CHECK-NEXT: lookupSysRegByName(StringRef Name);
// CHECK-NEXT: #endif

// CHECK: #ifdef GET_List_IMPL
// CHECK-NEXT: constexpr SysReg List[] = {
// CHECK-NEXT:   { "csr1", 0x7C0,  {Feature1}  }, // 0
// CHECK-NEXT:   { "csr2", 0x7C0,  {Feature2}  }, // 1
// CHECK-NEXT:   { "aabbc", 0x7C1,  {Feature1}  }, // 2
// CHECK-NEXT:   { "aaabb", 0x7C2,  {Feature2}  }, // 3
// CHECK-NEXT:  };
// CHECK-NEXT: constexpr SysReg ListforlookupSysRegByName[] = {
// CHECK-NEXT:   { "aaabb", 0x7C2,  {Feature2}  },
// CHECK-NEXT:   { "aabbc", 0x7C1,  {Feature1}  },
// CHECK-NEXT:   { "csr1", 0x7C0,  {Feature1}  },
// CHECK-NEXT: { "csr2", 0x7C0,  {Feature2}  },
// CHECK-NEXT:  };

// CHECK: llvm::iterator_range<const SysReg *> 
// CHECK-NEXT: lookupSysRegByEncoding(uint16_t Encoding) {
// CHECK-NEXT:   SysReg Key;
// CHECK-NEXT:   Key.Encoding = Encoding;
// CHECK-NEXT:   auto Table = ArrayRef(List);
// CHECK-NEXT:   auto It = std::equal_range(Table.begin(), Table.end(), Key,
// CHECK-NEXT:     [](const SysReg &LHS, const SysReg &RHS) {
// CHECK-NEXT:       if (LHS.Encoding < RHS.Encoding)
// CHECK-NEXT:         return true;
// CHECK-NEXT:       if (LHS.Encoding > RHS.Encoding)
// CHECK-NEXT:         return false;
// CHECK-NEXT:       return false;
// CHECK-NEXT:     });

// CHECK:   return llvm::make_range(It.first, It.second);
// CHECK-NEXT: }

// CHECK: llvm::iterator_range<const SysReg *>
// CHECK-NEXT: lookupSysRegByName(StringRef Name) {
// CHECK-NEXT:  SysReg Key;
// CHECK-NEXT:  Key.Name = Name.data();
// CHECK-NEXT:  auto Table = ArrayRef(ListforlookupSysRegByName);
// CHECK-NEXT:  auto It = std::equal_range(Table.begin(), Table.end(), Key,
// CHECK-NEXT:    [](const SysReg &LHS, const SysReg &RHS) {
// CHECK-NEXT:      int CmpName = StringRef(LHS.Name).compare(RHS.Name);
// CHECK-NEXT:      if (CmpName < 0) return true;
// CHECK-NEXT:      if (CmpName > 0) return false;
// CHECK-NEXT:      return false;
// CHECK-NEXT:    });

// CHECK:   return llvm::make_range(It.first, It.second);
// CHECK-NEXT: }
// CHECK-NEXT: #endif

// CHECK: #undef GET_List_DECL
// CHECK-NEXT: #undef GET_List_IMPL
