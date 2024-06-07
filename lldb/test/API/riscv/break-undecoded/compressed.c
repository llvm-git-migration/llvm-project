int main() {
  // This instruction is not valid, but we have an ability to set
  // software breakpoint.
  // This results illegal instruction during execution, not fail to set
  // breakpoint
  asm volatile(".2byte 0xaf");
}
