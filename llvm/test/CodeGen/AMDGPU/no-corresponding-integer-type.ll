; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 %s -o -

define void @no_corresponding_integer_type(i8 %arg, ptr addrspace(1) %ptr) {
entry:
  %load = load <3 x i8>, ptr addrspace(1) %ptr, align 1
  %elt0 = extractelement <3 x i8> %load, i64 0
  %mul0 = mul i8 %elt0, %arg
  %or = or i8 %mul0, 1
  %mul1 = mul i8 %arg, %arg
  %add = add i8 %mul1, %or
  store i8 %add, ptr addrspace(1) %ptr, align 1
  ret void
}
