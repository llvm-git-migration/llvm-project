! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
! UNSUPPORTED: sparc-target-arch

! CHECK-LABEL: c.func @_QQmain
program s
  use ieee_arithmetic

  use ieee_exceptions

  ! CHECK:     %[[V_56:[0-9]+]] = fir.address_of(@_QFEstatus) : !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_status_type{_QM__fortran_ieee_exceptionsTieee_status_type.__data:!fir.array<8xi32>,_QM__fortran_ieee_exceptionsTieee_status_type.__allocatable_data:!fir.box<!fir.heap<!fir.array<?xi8>>>}>>
  ! CHECK:     %[[V_57:[0-9]+]]:2 = hlfir.declare %[[V_56]] {uniq_name = "_QFEstatus"} : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_status_type{_QM__fortran_ieee_exceptionsTieee_status_type.__data:!fir.array<8xi32>,_QM__fortran_ieee_exceptionsTieee_status_type.__allocatable_data:!fir.box<!fir.heap<!fir.array<?xi8>>>}>>) -> (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_status_type{_QM__fortran_ieee_exceptionsTieee_status_type.__data:!fir.array<8xi32>,_QM__fortran_ieee_exceptionsTieee_status_type.__allocatable_data:!fir.box<!fir.heap<!fir.array<?xi8>>>}>>, !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_status_type{_QM__fortran_ieee_exceptionsTieee_status_type.__data:!fir.array<8xi32>,_QM__fortran_ieee_exceptionsTieee_status_type.__allocatable_data:!fir.box<!fir.heap<!fir.array<?xi8>>>}>>)
  type(ieee_status_type) :: status

  ! CHECK:     %[[V_58:[0-9]+]] = fir.alloca !fir.array<5x!fir.logical<4>> {bindc_name = "v", uniq_name = "_QFEv"}
  ! CHECK:     %[[V_59:[0-9]+]] = fir.shape %c5{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_60:[0-9]+]]:2 = hlfir.declare %[[V_58]](%[[V_59]]) {uniq_name = "_QFEv"} : (!fir.ref<!fir.array<5x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5x!fir.logical<4>>>, !fir.ref<!fir.array<5x!fir.logical<4>>>)
  logical :: v(size(ieee_all))

  ! CHECK:     %[[V_61:[0-9]+]] = fir.address_of(@_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.0)
  ! CHECK:     %[[V_62:[0-9]+]] = fir.shape %c5{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_63:[0-9]+]]:2 = hlfir.declare %[[V_61]](%[[V_62]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.0"}
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c5{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_119:[0-9]+]] = hlfir.designate %[[V_63]]#0 (%arg0)
  ! CHECK:       %[[V_120:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_flag_type.flag, !fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>
  ! CHECK:       %[[V_121:[0-9]+]] = fir.coordinate_of %[[V_119]], %[[V_120]]
  ! CHECK:       %[[V_122:[0-9]+]] = fir.load %[[V_121]] : !fir.ref<i8>
  ! CHECK:       %[[V_123:[0-9]+]] = fir.convert %[[V_122]] : (i8) -> i32
  ! CHECK:       %[[V_124:[0-9]+]] = fir.call @_FortranAMapException(%[[V_123]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_125:[0-9]+]] = fir.convert %true{{[_0-9]*}} : (i1) -> i1
  ! CHECK:       fir.if %[[V_125]] {
  ! CHECK:         %[[V_126:[0-9]+]] = fir.call @feenableexcept(%[[V_124]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_126:[0-9]+]] = fir.call @fedisableexcept(%[[V_124]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_halting_mode(ieee_all, .true.)

  ! CHECK:     %[[V_64:[0-9]+]] = fir.address_of(@_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.0)
  ! CHECK:     %[[V_65:[0-9]+]] = fir.shape %c5{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_66:[0-9]+]]:2 = hlfir.declare %[[V_64]](%[[V_65]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.0"}
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c5{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_119:[0-9]+]] = hlfir.designate %[[V_66]]#0 (%arg0)
  ! CHECK:       %[[V_120:[0-9]+]] = hlfir.designate %[[V_60]]#0 (%arg0) : (!fir.ref<!fir.array<5x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_121:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_flag_type.flag, !fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>
  ! CHECK:       %[[V_122:[0-9]+]] = fir.coordinate_of %[[V_119]], %[[V_121]]
  ! CHECK:       %[[V_123:[0-9]+]] = fir.load %[[V_122]] : !fir.ref<i8>
  ! CHECK:       %[[V_124:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:       %[[V_125:[0-9]+]] = fir.convert %[[V_123]] : (i8) -> i32
  ! CHECK:       %[[V_126:[0-9]+]] = fir.call @_FortranAMapException(%[[V_125]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_127:[0-9]+]] = arith.andi %[[V_124]], %[[V_126]] : i32
  ! CHECK:       %[[V_128:[0-9]+]] = arith.cmpi ne, %[[V_127]], %c0{{.*}} : i32
  ! CHECK:       %[[V_129:[0-9]+]] = fir.convert %[[V_128]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_129]] to %[[V_120]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_halting_mode(ieee_all, v)

  print*, 'halting_mode [T T T T T] :', v

  ! CHECK:     %[[V_80:[0-9]+]] = fir.convert %[[V_57]]#1 : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_status_type{_QM__fortran_ieee_exceptionsTieee_status_type.__data:!fir.array<8xi32>,_QM__fortran_ieee_exceptionsTieee_status_type.__allocatable_data:!fir.box<!fir.heap<!fir.array<?xi8>>>}>>) -> !fir.ref<i32>
  ! CHECK:     %[[V_81:[0-9]+]] = fir.call @fegetenv(%[[V_80]]) fastmath<contract> : (!fir.ref<i32>) -> i32
  call ieee_get_status(status)

  ! CHECK:     %[[V_82:[0-9]+]] = fir.address_of(@_QQro.3x_QM__fortran_builtinsT__builtin_ieee_flag_type.1)
  ! CHECK:     %[[V_83:[0-9]+]] = fir.shape %c3{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_84:[0-9]+]]:2 = hlfir.declare %[[V_82]](%[[V_83]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x_QM__fortran_builtinsT__builtin_ieee_flag_type.1"}
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c3{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_119:[0-9]+]] = hlfir.designate %[[V_84]]#0 (%arg0)
  ! CHECK:       %[[V_120:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_flag_type.flag, !fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>
  ! CHECK:       %[[V_121:[0-9]+]] = fir.coordinate_of %[[V_119]], %[[V_120]]
  ! CHECK:       %[[V_122:[0-9]+]] = fir.load %[[V_121]] : !fir.ref<i8>
  ! CHECK:       %[[V_123:[0-9]+]] = fir.convert %[[V_122]] : (i8) -> i32
  ! CHECK:       %[[V_124:[0-9]+]] = fir.call @_FortranAMapException(%[[V_123]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_125:[0-9]+]] = fir.convert %false{{[_0-9]*}} : (i1) -> i1
  ! CHECK:       fir.if %[[V_125]] {
  ! CHECK:         %[[V_126:[0-9]+]] = fir.call @feenableexcept(%[[V_124]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_126:[0-9]+]] = fir.call @fedisableexcept(%[[V_124]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_halting_mode(ieee_usual, .false.)

  ! CHECK:     %[[V_85:[0-9]+]] = fir.address_of(@_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.0)
  ! CHECK:     %[[V_86:[0-9]+]] = fir.shape %c5{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_87:[0-9]+]]:2 = hlfir.declare %[[V_85]](%[[V_86]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.0"}
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c5{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_119:[0-9]+]] = hlfir.designate %[[V_87]]#0 (%arg0)
  ! CHECK:       %[[V_120:[0-9]+]] = hlfir.designate %[[V_60]]#0 (%arg0) : (!fir.ref<!fir.array<5x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_121:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_flag_type.flag, !fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>
  ! CHECK:       %[[V_122:[0-9]+]] = fir.coordinate_of %[[V_119]], %[[V_121]]
  ! CHECK:       %[[V_123:[0-9]+]] = fir.load %[[V_122]] : !fir.ref<i8>
  ! CHECK:       %[[V_124:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:       %[[V_125:[0-9]+]] = fir.convert %[[V_123]] : (i8) -> i32
  ! CHECK:       %[[V_126:[0-9]+]] = fir.call @_FortranAMapException(%[[V_125]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_127:[0-9]+]] = arith.andi %[[V_124]], %[[V_126]] : i32
  ! CHECK:       %[[V_128:[0-9]+]] = arith.cmpi ne, %[[V_127]], %c0{{.*}} : i32
  ! CHECK:       %[[V_129:[0-9]+]] = fir.convert %[[V_128]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_129]] to %[[V_120]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_halting_mode(ieee_all, v)

  print*, 'halting_mode [F F F T T] :', v

  ! CHECK:     %[[V_101:[0-9]+]] = fir.convert %[[V_57]]#1 : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_status_type{_QM__fortran_ieee_exceptionsTieee_status_type.__data:!fir.array<8xi32>,_QM__fortran_ieee_exceptionsTieee_status_type.__allocatable_data:!fir.box<!fir.heap<!fir.array<?xi8>>>}>>) -> !fir.ref<i32>
  ! CHECK:     %[[V_102:[0-9]+]] = fir.call @fesetenv(%[[V_101]]) fastmath<contract> : (!fir.ref<i32>) -> i32
  call ieee_set_status(status)

  ! CHECK:     %[[V_103:[0-9]+]] = fir.address_of(@_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.0)
  ! CHECK:     %[[V_104:[0-9]+]] = fir.shape %c5{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_105:[0-9]+]]:2 = hlfir.declare %[[V_103]](%[[V_104]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.0"}
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c5{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_119:[0-9]+]] = hlfir.designate %[[V_105]]#0 (%arg0)
  ! CHECK:       %[[V_120:[0-9]+]] = hlfir.designate %[[V_60]]#0 (%arg0) : (!fir.ref<!fir.array<5x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_121:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_flag_type.flag, !fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>
  ! CHECK:       %[[V_122:[0-9]+]] = fir.coordinate_of %[[V_119]], %[[V_121]]
  ! CHECK:       %[[V_123:[0-9]+]] = fir.load %[[V_122]] : !fir.ref<i8>
  ! CHECK:       %[[V_124:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:       %[[V_125:[0-9]+]] = fir.convert %[[V_123]] : (i8) -> i32
  ! CHECK:       %[[V_126:[0-9]+]] = fir.call @_FortranAMapException(%[[V_125]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_127:[0-9]+]] = arith.andi %[[V_124]], %[[V_126]] : i32
  ! CHECK:       %[[V_128:[0-9]+]] = arith.cmpi ne, %[[V_127]], %c0{{.*}} : i32
  ! CHECK:       %[[V_129:[0-9]+]] = fir.convert %[[V_128]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_129]] to %[[V_120]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_halting_mode(ieee_all, v)

  print*, 'halting_mode [T T T T T] :', v
end
