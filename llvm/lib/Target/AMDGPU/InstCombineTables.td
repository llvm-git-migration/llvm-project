include "llvm/TableGen/SearchableTable.td"
include "llvm/Target/Target.td"

// Dummy target to pacify SearchTableEmitter code.
def DummyTarget : Target;

def AMDGPUImageDMaskIntrinsicTable : GenericTable {
  let FilterClass = "AMDGPUImageDMaskIntrinsic";
  let Fields = ["Intr"];

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "getAMDGPUImageDMaskIntrinsic";
  let PrimaryKeyEarlyOut = 1;
}
