// RUN: %check_clang_tidy -std=c++11 %s modernize-use-cpp-style-comments %t

static auto PI = 3.14159265; /* value of pi upto 8 decimal places */
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: use C++ style comments '//' instead of C style comments '/*...*/' [modernize-use-cpp-style-comments]

/******************************************************
* Fancy  frame comment goes here
*******************************************************/
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++ style comments '//' instead of C style comments '/*...*/' [modernize-use-cpp-style-comments]

/** \brief Brief description.
 *         Brief description continued.
 *
 *  Detailed description starts here. (this is a doxygen comment)
 */
// CHECK-MESSAGES: :[[@LINE-5]]:1: warning: use C++ style comments '//' instead of C style comments '/*...*/' [modernize-use-cpp-style-comments]
