// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

emitc.global extern @decl : i8
// CPP-DEFAULT: extern int8_t decl;
// CPP-DECLTOP: extern int8_t decl;

emitc.global @uninit : i32
// CPP-DEFAULT: int32_t uninit;
// CPP-DECLTOP: int32_t uninit;

emitc.global @myglobal_int : i32 = 4
// CPP-DEFAULT: int32_t myglobal_int = 4;
// CPP-DECLTOP: int32_t myglobal_int = 4;

emitc.global @myglobal : !emitc.array<2xf32> = dense<4.000000e+00>
// CPP-DEFAULT: float myglobal[2] = {4.000000000e+00f, 4.000000000e+00f};
// CPP-DECLTOP: float myglobal[2] = {4.000000000e+00f, 4.000000000e+00f};

emitc.global const @myconstant : !emitc.array<2xi16> = dense<2>
// CPP-DEFAULT: const int16_t myconstant[2] = {2, 2};
// CPP-DECLTOP: const int16_t myconstant[2] = {2, 2};

emitc.global extern const @extern_constant : !emitc.array<2xi16>
// CPP-DEFAULT: extern const int16_t extern_constant[2];
// CPP-DECLTOP: extern const int16_t extern_constant[2];

emitc.global static @static_var : f32
// CPP-DEFAULT: static float static_var;
// CPP-DECLTOP: static float static_var;

emitc.global static @static_const : f32 = 3.0
// CPP-DEFAULT: static float static_const = 3.000000000e+00f;
// CPP-DECLTOP: static float static_const = 3.000000000e+00f;

emitc.global @opaque_init : !emitc.opaque<"char"> = #emitc.opaque<"CHAR_MIN">
// CPP-DEFAULT: char opaque_init = CHAR_MIN;
// CPP-DECLTOP: char opaque_init = CHAR_MIN;

func.func @use_global(%i: index) -> f32 {
  %0 = emitc.get_global @myglobal : !emitc.array<2xf32>
  %1 = emitc.subscript %0[%i] : (!emitc.array<2xf32>, index) -> !emitc.lvalue<f32>
  %2 = emitc.lvalue_load %1 : <f32>
  return %2 : f32
}
// CPP-DEFAULT-LABEL: use_global
// CPP-DEFAULT-SAME: (size_t [[V1:.*]])
// CPP-DEFAULT-NEXT: float [[V2:[^ ]*]] = myglobal[[[V1]]];
// CPP-DEFAULT-NEXT: return [[V2]];

// CPP-DECLTOP-LABEL: use_global
// CPP-DECLTOP-SAME: (size_t [[V1:.*]])
// CPP-DECLTOP-NEXT: float [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V2]] = myglobal[[[V1]]];
// CPP-DECLTOP-NEXT: return [[V2]];
