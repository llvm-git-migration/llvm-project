#include <string>

int main() {
  const char *stack_string_ptr = "there_is_only_one_of_me";
  (void)stack_string_ptr;
  const std::string heap_string1("there_is_exactly_two_of_me");
  const std::string heap_string2("there_is_exactly_two_of_me");
  heap_string1.data(); // force instantiation of string::data()

  return 0; // break here
}
