! RUN: %python %S/test_folding.py %s %flang_fc1 -funsigned
! UNSIGNED operations and intrinsic functions

module m

  logical, parameter :: test_huge_1  = huge(0u_1)  == 255u_1
  logical, parameter :: test_huge_2  = huge(0u_2)  == 65535u_2
  logical, parameter :: test_huge_4  = huge(0u_4)  == uint(huge(0_4),4) * 2u + 1u
  logical, parameter :: test_huge_8  = huge(0u_8)  == uint(huge(0_8),8) * 2u + 1u
  logical, parameter :: test_huge_16 = huge(0u_16) == uint(huge(0_16),16) * 2u + 1u

end
