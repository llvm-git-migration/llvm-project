#include <cstring>

int main() {
  constexpr char SINGLE_INSTANCE_STRING[] = "there_is_only_one_of_me";
  constexpr size_t single_instance_size = sizeof(SINGLE_INSTANCE_STRING) + 1;
  char *single_instance = new char[single_instance_size];
  strcpy(single_instance, SINGLE_INSTANCE_STRING);

  constexpr char DOUBLE_INSTANCE_STRING[] = "there_is_exactly_two_of_me";
  constexpr size_t double_instance_size = sizeof(DOUBLE_INSTANCE_STRING) + 1;
  char *double_instance = new char[double_instance_size];
  char *double_instance_copy = new char[double_instance_size];
  strcpy(double_instance, DOUBLE_INSTANCE_STRING);
  strcpy(double_instance_copy, DOUBLE_INSTANCE_STRING);

  delete[] single_instance; // break here
  delete[] double_instance;
  delete[] double_instance_copy;

  return 0;
}
