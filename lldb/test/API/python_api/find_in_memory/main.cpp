#include <string>

int main() {
  const char *stack_pointer = "stack_there_is_only_one_of_me";
  (void)stack_pointer;
  const std::string heap_string1("heap_there_is_exactly_two_of_me");
  const std::string heap_string2("heap_there_is_exactly_two_of_me");
  const char *heap_pointer1 = heap_string1.data();
  const char *heap_pointer2 = heap_string2.data();
  (void)heap_pointer1;
  (void)heap_pointer2; // break here
  return 0;
}
