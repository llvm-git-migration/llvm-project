! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
! UNSUPPORTED: sparc-target-arch

! CHECK-LABEL: c.func @_QQmain
program m
  use ieee_arithmetic
  use ieee_exceptions

  ! CHECK:     %[[V_59:[0-9]+]] = fir.address_of(@_QFEmodes) : !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>,_QM__fortran_ieee_exceptionsTieee_modes_type.__allocatable_data:!fir.box<!fir.heap<!fir.array<?xi8>>>}>>
  ! CHECK:     %[[V_60:[0-9]+]]:2 = hlfir.declare %[[V_59]] {uniq_name = "_QFEmodes"} : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>,_QM__fortran_ieee_exceptionsTieee_modes_type.__allocatable_data:!fir.box<!fir.heap<!fir.array<?xi8>>>}>>) -> (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>,_QM__fortran_ieee_exceptionsTieee_modes_type.__allocatable_data:!fir.box<!fir.heap<!fir.array<?xi8>>>}>>, !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>,_QM__fortran_ieee_exceptionsTieee_modes_type.__allocatable_data:!fir.box<!fir.heap<!fir.array<?xi8>>>}>>)
  type(ieee_modes_type) :: modes

  ! CHECK:     %[[V_61:[0-9]+]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}> {bindc_name = "round", uniq_name = "_QFEround"}
  ! CHECK:     %[[V_62:[0-9]+]]:2 = hlfir.declare %[[V_61]] {uniq_name = "_QFEround"}
  type(ieee_round_type) :: round

  ! CHECK:     %[[V_68:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.0)
  ! CHECK:     %[[V_69:[0-9]+]]:2 = hlfir.declare %[[V_68]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.0"}
  ! CHECK:     %[[V_70:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
  ! CHECK:     %[[V_71:[0-9]+]] = fir.coordinate_of %[[V_69]]#1, %[[V_70]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_72:[0-9]+]] = fir.load %[[V_71]] : !fir.ref<i8>
  ! CHECK:     %[[V_73:[0-9]+]] = fir.convert %[[V_72]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_73]]) fastmath<contract> : (i32) -> ()
  call ieee_set_rounding_mode(ieee_up)

  ! CHECK:     %[[V_74:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
  ! CHECK:     %[[V_75:[0-9]+]] = fir.coordinate_of %[[V_62]]#1, %[[V_74]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_76:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_77:[0-9]+]] = fir.convert %[[V_76]] : (i32) -> i8
  ! CHECK:     fir.store %[[V_77]] to %[[V_75]] : !fir.ref<i8>
  call ieee_get_rounding_mode(round)

  print*, 'rounding_mode [up     ] : ', mode_name(round)

  ! CHECK:     %[[V_98:[0-9]+]] = fir.convert %[[V_60]]#1 : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>,_QM__fortran_ieee_exceptionsTieee_modes_type.__allocatable_data:!fir.box<!fir.heap<!fir.array<?xi8>>>}>>) -> !fir.ref<i32>
  ! CHECK:     %[[V_99:[0-9]+]] = fir.call @fegetmode(%[[V_98]]) fastmath<contract> : (!fir.ref<i32>) -> i32
  call ieee_get_modes(modes)

  ! CHECK:     %[[V_100:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.1)
  ! CHECK:     %[[V_101:[0-9]+]]:2 = hlfir.declare %[[V_100]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.1"}
  ! CHECK:     %[[V_102:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
  ! CHECK:     %[[V_103:[0-9]+]] = fir.coordinate_of %[[V_101]]#1, %[[V_102]]
  ! CHECK:     %[[V_104:[0-9]+]] = fir.load %[[V_103]] : !fir.ref<i8>
  ! CHECK:     %[[V_105:[0-9]+]] = fir.convert %[[V_104]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_105]]) fastmath<contract> : (i32) -> ()
  call ieee_set_rounding_mode(ieee_to_zero)

  ! CHECK:     %[[V_106:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
  ! CHECK:     %[[V_107:[0-9]+]] = fir.coordinate_of %[[V_62]]#1, %[[V_106]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_108:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_109:[0-9]+]] = fir.convert %[[V_108]] : (i32) -> i8
  ! CHECK:     fir.store %[[V_109]] to %[[V_107]] : !fir.ref<i8>
  call ieee_get_rounding_mode(round)

  print*, 'rounding_mode [to_zero] : ', mode_name(round)

  ! CHECK:     %[[V_130:[0-9]+]] = fir.convert %[[V_60]]#1 : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>,_QM__fortran_ieee_exceptionsTieee_modes_type.__allocatable_data:!fir.box<!fir.heap<!fir.array<?xi8>>>}>>) -> !fir.ref<i32>
  ! CHECK:     %[[V_131:[0-9]+]] = fir.call @fesetmode(%[[V_130]]) fastmath<contract> : (!fir.ref<i32>) -> i32
  call ieee_set_modes(modes)

  ! CHECK:     %[[V_132:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
  ! CHECK:     %[[V_133:[0-9]+]] = fir.coordinate_of %[[V_62]]#1, %[[V_132]]
  ! CHECK:     %[[V_134:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_135:[0-9]+]] = fir.convert %[[V_134]] : (i32) -> i8
  ! CHECK:     fir.store %[[V_135]] to %[[V_133]] : !fir.ref<i8>
  call ieee_get_rounding_mode(round)

  print*, 'rounding_mode [up     ] : ', mode_name(round)

contains
  character(7) function mode_name(m)
    type(ieee_round_type), intent(in) :: m
    if (m == ieee_nearest) then
      mode_name = 'nearest'
    else if (m == ieee_to_zero) then
      mode_name = 'to_zero'
    else if (m == ieee_up) then
      mode_name = 'up'
    else if (m == ieee_down) then
      mode_name = 'down'
    else if (m == ieee_away) then
      mode_name = 'away'
    else if (m == ieee_other) then
      mode_name = 'other'
    else
      mode_name = '???'
    endif
  end
end
