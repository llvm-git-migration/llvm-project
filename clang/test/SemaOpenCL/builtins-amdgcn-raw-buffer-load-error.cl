// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu verde -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

typedef char i8;
typedef short i16;
typedef int i32;
typedef int i64 __attribute__((ext_vector_type(2)));
typedef int i96 __attribute__((ext_vector_type(3)));
typedef int i128 __attribute__((ext_vector_type(4)));

i8 test_amdgcn_raw_ptr_buffer_load_b8(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  return __builtin_amdgcn_raw_buffer_load_b8(rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_load_b8' must be a constant integer}}
}

i16 test_amdgcn_raw_ptr_buffer_load_b16(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  return __builtin_amdgcn_raw_buffer_load_b16(rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_load_b16' must be a constant integer}}
}

i32 test_amdgcn_raw_ptr_buffer_load_b32(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  return __builtin_amdgcn_raw_buffer_load_b32(rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_load_b32' must be a constant integer}}
}

i64 test_amdgcn_raw_ptr_buffer_load_b64(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  return __builtin_amdgcn_raw_buffer_load_b64(rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_load_b64' must be a constant integer}}
}

i96 test_amdgcn_raw_ptr_buffer_load_b96(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  return __builtin_amdgcn_raw_buffer_load_b96(rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_load_b96' must be a constant integer}}
}

i128 test_amdgcn_raw_ptr_buffer_load_b128(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  return __builtin_amdgcn_raw_buffer_load_b128(rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_load_b128' must be a constant integer}}
}
