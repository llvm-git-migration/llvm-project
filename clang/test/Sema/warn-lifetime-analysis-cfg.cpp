// RUN: %clang_cc1 -Wno-dangling -Wdangling-cfg -verify -std=c++11 %s
#include "Inputs/lifetime-analysis.h"

std::string_view unknown(std::string_view s);
std::string_view unknown();
std::string temporary();

std::string_view simple_with_ctor() {
  std::string s1; // expected-note {{reference to this stack variable is returned}}
  std::string_view cs1 = s1;
  return cs1; // expected-warning {{returning reference to a stack variable}}
}

std::string_view simple_with_assignment() {
  std::string s1; // expected-note {{reference to this stack variable is returned}}
  std::string_view cs1;
  cs1 = s1;
  return cs1; // expected-warning {{returning reference to a stack variable}}
}

std::string_view use_unknown() {
  std::string s1;
  std::string_view cs1 = s1;
  cs1 = unknown();
  return cs1; // ok.
}

std::string_view overwrite_unknown() {
  std::string s1; // expected-note {{reference to this stack variable is returned}}
  std::string_view cs1 = s1;
  cs1 = unknown();
  cs1 = s1;
  return cs1; // expected-warning {{returning reference to a stack variable}}
}

std::string_view overwrite_with_unknown() {
  std::string s1;
  std::string_view cs1 = s1;
  cs1 = s1;
  cs1 = unknown();
  return cs1; // Ok.
}

std::string_view return_unknown() {
  std::string s1;
  std::string_view cs1 = s1;
  return unknown(cs1);
}

std::string_view multiple_assignments() {
  std::string s1;
  std::string s2 = s1; // expected-note {{reference to this stack variable is returned}}
  std::string_view cs1 = s1;
  std::string_view cs2 = s1;
  s2 = s1; // Ignore owner transfers.
  cs1 = s1;
  cs2 = s2;
  cs1 = cs2;
  cs2 = s1;
  return cs1; // expected-warning {{returning reference to a stack variable}}
}

std::string_view global_view;
std::string_view ignore_global_view() {
  std::string s;
  global_view = s; // TODO: We can still track writes to static storage!
  return global_view; // Ok.
}
std::string global_owner;
std::string_view ignore_global_owner() {
  std::string_view sv;
  sv = global_owner;
  return sv; // Ok.
}

std::string_view return_temporary() {
  return 
    temporary(); // expected-warning {{returning reference to a temporary object}}
}

std::string_view store_temporary() {
  std::string_view sv =
    temporary(); // expected-warning {{returning reference to a temporary object}}
  return sv;
}

std::string_view return_local() {
  std::string local; // expected-note {{reference to this stack variable is returned}}
  return // expected-warning {{returning reference to a stack variable}}
    local; 
}

// Parameters
std::string_view params_owner_and_views(
  std::string_view sv, 
  std::string s) { // expected-note {{reference to this stack variable is returned}}
  sv = s;
  return sv; // expected-warning {{returning reference to a stack variable}}
}

std::string_view param_owner(std::string s) { // expected-note {{reference to this stack variable is returned}}
  return s; // expected-warning {{returning reference to a stack variable}}
}
std::string_view param_owner_ref(const std::string& s) {
  return s;
}

std::string& get_str_ref();
std::string_view ref_to_global_str() {
  const std::string& s =  get_str_ref();
  std::string_view sv = s;
  return sv;
}
std::string_view view_to_global_str() {
  std::string_view s =  get_str_ref();
  std::string_view sv = s;
  sv = s;
  return sv;
}
std::string_view copy_of_global_str() {
  std::string s =  get_str_ref(); // expected-note {{reference to this stack variable is returned}}
  std::string_view sv = s;
  return sv; // expected-warning {{returning reference to a stack variable}}
}

// TODO: Use lifetimebound in function calls. Use Pointer to Owner of pointer
// std::string_view containerOfString() {
//   std::vector<std::string> local;
//   return local.at(0);
// }

// std::string_view containerOfViewMultistmt() {
//   std::vector<std::string> local;
//   std::string_view sv = local.at(0);
//   return sv;
// }
