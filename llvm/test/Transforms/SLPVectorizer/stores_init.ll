define void @stores_init(ptr %ptr) {
  %gep0 = getelementptr i8, ptr %ptr, i64 0
  %gep1 = getelementptr i8, ptr %ptr, i64 1
  %gep2 = getelementptr i8, ptr %ptr, i64 2
  %gep3 = getelementptr i8, ptr %ptr, i64 3
  store i8 0, ptr %gep0
  store i8 1, ptr %gep1
  store i16 2, ptr %gep2
  ret void
}
