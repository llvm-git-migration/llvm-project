//===-- Interface for malloc status 00-------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_STATUS_H
#define LLVM_LIBC_SRC_STDLIB_STATUS_H

#include "src/__support/CPP/type_traits.h"

// This is the Status enum. Status is used to return the status from an
// operation.
//
// In C++, use the Status class instead of the Status enum. Status and
// Status implicitly convert to one another and can be passed cleanly between C
// and C++ APIs.
//
// Status uses the canonical Google error codes. The following enum was based
// on Abseil's status/status.h.
typedef enum {
  STATUS_OK = 0,                  // Use OkStatus() in C++
  STATUS_CANCELLED = 1,           // Use Status::Cancelled() in C++
  STATUS_UNKNOWN = 2,             // Use Status::Unknown() in C++
  STATUS_INVALID_ARGUMENT = 3,    // Use Status::InvalidArgument() in C++
  STATUS_DEADLINE_EXCEEDED = 4,   // Use Status::DeadlineExceeded() in C++
  STATUS_NOT_FOUND = 5,           // Use Status::NotFound() in C++
  STATUS_ALREADY_EXISTS = 6,      // Use Status::AlreadyExists() in C++
  STATUS_PERMISSION_DENIED = 7,   // Use Status::PermissionDenied() in C++
  STATUS_RESOURCE_EXHAUSTED = 8,  // Use Status::ResourceExhausted() in C++
  STATUS_FAILED_PRECONDITION = 9, // Use Status::FailedPrecondition() in C++
  STATUS_ABORTED = 10,            // Use Status::Aborted() in C++
  STATUS_OUT_OF_RANGE = 11,       // Use Status::OutOfRange() in C++
  STATUS_UNIMPLEMENTED = 12,      // Use Status::Unimplemented() in C++
  STATUS_INTERNAL = 13,           // Use Status::Internal() in C++
  STATUS_UNAVAILABLE = 14,        // Use Status::Unavailable() in C++
  STATUS_DATA_LOSS = 15,          // Use Status::DataLoss() in C++
  STATUS_UNAUTHENTICATED = 16,    // Use Status::Unauthenticated() in C++

  // NOTE: this error code entry should not be used and you should not rely on
  // its value, which may change.
  //
  // The purpose of this enumerated value is to force people who handle status
  // codes with `switch()` statements to *not* simply enumerate all possible
  // values, but instead provide a "default:" case. Providing such a default
  // case ensures that code will compile when new codes are added.
  STATUS_DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_,
} StatusKind; // Use Status in C++

/// `Status` is a thin, zero-cost abstraction around the `Status` enum. It
/// initializes to @status{OK} by default and adds `ok()` and `str()`
/// methods. Implicit conversions are permitted between `Status` and
/// `Status`.
///
/// An @status{OK} `Status` is created by the @cpp_func{OkStatus}
/// function or by the default `Status` constructor.  Non-OK `Status` is created
/// with a static member function that corresponds with the status code.
class Status {
public:
  using Code = StatusKind;

  // Functions that create a Status with the specified code.
  [[nodiscard]] static constexpr Status Cancelled() { return STATUS_CANCELLED; }
  [[nodiscard]] static constexpr Status Unknown() { return STATUS_UNKNOWN; }
  [[nodiscard]] static constexpr Status InvalidArgument() {
    return STATUS_INVALID_ARGUMENT;
  }
  [[nodiscard]] static constexpr Status DeadlineExceeded() {
    return STATUS_DEADLINE_EXCEEDED;
  }
  [[nodiscard]] static constexpr Status NotFound() { return STATUS_NOT_FOUND; }
  [[nodiscard]] static constexpr Status AlreadyExists() {
    return STATUS_ALREADY_EXISTS;
  }
  [[nodiscard]] static constexpr Status PermissionDenied() {
    return STATUS_PERMISSION_DENIED;
  }
  [[nodiscard]] static constexpr Status ResourceExhausted() {
    return STATUS_RESOURCE_EXHAUSTED;
  }
  [[nodiscard]] static constexpr Status FailedPrecondition() {
    return STATUS_FAILED_PRECONDITION;
  }
  [[nodiscard]] static constexpr Status Aborted() { return STATUS_ABORTED; }
  [[nodiscard]] static constexpr Status OutOfRange() {
    return STATUS_OUT_OF_RANGE;
  }
  [[nodiscard]] static constexpr Status Unimplemented() {
    return STATUS_UNIMPLEMENTED;
  }
  [[nodiscard]] static constexpr Status Internal() { return STATUS_INTERNAL; }
  [[nodiscard]] static constexpr Status Unavailable() {
    return STATUS_UNAVAILABLE;
  }
  [[nodiscard]] static constexpr Status DataLoss() { return STATUS_DATA_LOSS; }
  [[nodiscard]] static constexpr Status Unauthenticated() {
    return STATUS_UNAUTHENTICATED;
  }
  // clang-format on

  // Statuses are created with a Status::Code.
  constexpr Status(Code code = STATUS_OK) : code_(code) {}

  constexpr Status(const Status &) = default;
  constexpr Status &operator=(const Status &) = default;

  /// Returns the `Status::Code` (`Status`) for this `Status`.
  constexpr Code code() const { return code_; }

  /// True if the status is @status{OK}.
  ///
  /// This function is provided in place of an `IsOk()` function.
  [[nodiscard]] constexpr bool ok() const { return code_ == STATUS_OK; }

  // Functions for checking which status this is.
  [[nodiscard]] constexpr bool IsCancelled() const {
    return code_ == STATUS_CANCELLED;
  }
  [[nodiscard]] constexpr bool IsUnknown() const {
    return code_ == STATUS_UNKNOWN;
  }
  [[nodiscard]] constexpr bool IsInvalidArgument() const {
    return code_ == STATUS_INVALID_ARGUMENT;
  }
  [[nodiscard]] constexpr bool IsDeadlineExceeded() const {
    return code_ == STATUS_DEADLINE_EXCEEDED;
  }
  [[nodiscard]] constexpr bool IsNotFound() const {
    return code_ == STATUS_NOT_FOUND;
  }
  [[nodiscard]] constexpr bool IsAlreadyExists() const {
    return code_ == STATUS_ALREADY_EXISTS;
  }
  [[nodiscard]] constexpr bool IsPermissionDenied() const {
    return code_ == STATUS_PERMISSION_DENIED;
  }
  [[nodiscard]] constexpr bool IsResourceExhausted() const {
    return code_ == STATUS_RESOURCE_EXHAUSTED;
  }
  [[nodiscard]] constexpr bool IsFailedPrecondition() const {
    return code_ == STATUS_FAILED_PRECONDITION;
  }
  [[nodiscard]] constexpr bool IsAborted() const {
    return code_ == STATUS_ABORTED;
  }
  [[nodiscard]] constexpr bool IsOutOfRange() const {
    return code_ == STATUS_OUT_OF_RANGE;
  }
  [[nodiscard]] constexpr bool IsUnimplemented() const {
    return code_ == STATUS_UNIMPLEMENTED;
  }
  [[nodiscard]] constexpr bool IsInternal() const {
    return code_ == STATUS_INTERNAL;
  }
  [[nodiscard]] constexpr bool IsUnavailable() const {
    return code_ == STATUS_UNAVAILABLE;
  }
  [[nodiscard]] constexpr bool IsDataLoss() const {
    return code_ == STATUS_DATA_LOSS;
  }
  [[nodiscard]] constexpr bool IsUnauthenticated() const {
    return code_ == STATUS_UNAUTHENTICATED;
  }

  /// Updates this `Status` to the provided `Status` IF this status is
  /// @status{OK}. This is useful for tracking the first encountered error,
  /// as calls to this helper will not change one error status to another error
  /// status.
  constexpr void Update(Status other) {
    if (ok()) {
      code_ = other.code();
    }
  }

  /// Ignores any errors. This method does nothing except potentially suppress
  /// complaints from any tools that are checking that errors are not dropped on
  /// the floor.
  constexpr void IgnoreError() const {}

private:
  Code code_;
};

class StatusWithSize {
public:
  static constexpr StatusWithSize Cancelled(size_t size = 0) {
    return StatusWithSize(Status::Cancelled(), size);
  }
  static constexpr StatusWithSize Unknown(size_t size = 0) {
    return StatusWithSize(Status::Unknown(), size);
  }
  static constexpr StatusWithSize InvalidArgument(size_t size = 0) {
    return StatusWithSize(Status::InvalidArgument(), size);
  }
  static constexpr StatusWithSize DeadlineExceeded(size_t size = 0) {
    return StatusWithSize(Status::DeadlineExceeded(), size);
  }
  static constexpr StatusWithSize NotFound(size_t size = 0) {
    return StatusWithSize(Status::NotFound(), size);
  }
  static constexpr StatusWithSize AlreadyExists(size_t size = 0) {
    return StatusWithSize(Status::AlreadyExists(), size);
  }
  static constexpr StatusWithSize PermissionDenied(size_t size = 0) {
    return StatusWithSize(Status::PermissionDenied(), size);
  }
  static constexpr StatusWithSize Unauthenticated(size_t size = 0) {
    return StatusWithSize(Status::Unauthenticated(), size);
  }
  static constexpr StatusWithSize ResourceExhausted(size_t size = 0) {
    return StatusWithSize(Status::ResourceExhausted(), size);
  }
  static constexpr StatusWithSize FailedPrecondition(size_t size = 0) {
    return StatusWithSize(Status::FailedPrecondition(), size);
  }
  static constexpr StatusWithSize Aborted(size_t size = 0) {
    return StatusWithSize(Status::Aborted(), size);
  }
  static constexpr StatusWithSize OutOfRange(size_t size = 0) {
    return StatusWithSize(Status::OutOfRange(), size);
  }
  static constexpr StatusWithSize Unimplemented(size_t size = 0) {
    return StatusWithSize(Status::Unimplemented(), size);
  }
  static constexpr StatusWithSize Internal(size_t size = 0) {
    return StatusWithSize(Status::Internal(), size);
  }
  static constexpr StatusWithSize Unavailable(size_t size = 0) {
    return StatusWithSize(Status::Unavailable(), size);
  }
  static constexpr StatusWithSize DataLoss(size_t size = 0) {
    return StatusWithSize(Status::DataLoss(), size);
  }

  // Creates a StatusWithSize with OkStatus() and a size of 0.
  explicit constexpr StatusWithSize() : size_(0) {}

  // Creates a StatusWithSize with status OK and the provided size.
  // std::enable_if is used to prevent enum types (e.g. Status) from being used.
  // TODO(hepler): Add debug-only assert that size <= max_size().
  // template <typename T, typename =
  // LIBC_NAMESPACE::cpp::enable_if_t<std::is_integral<T>::value>> explicit
  // constexpr StatusWithSize(T size)
  //    : size_(static_cast<size_t>(size)) {}
  explicit constexpr StatusWithSize(size_t size)
      : size_(static_cast<size_t>(size)) {}

  // Creates a StatusWithSize with the provided status and size.
  explicit constexpr StatusWithSize(Status status, size_t size)
      : StatusWithSize((static_cast<size_t>(status.code()) << kStatusShift) |
                       size) {}

  constexpr StatusWithSize(const StatusWithSize &) = default;
  constexpr StatusWithSize &operator=(const StatusWithSize &) = default;

  /// ``Update`` s this status and adds to ``size``.
  ///
  /// The resulting ``StatusWithSize`` will have a size of
  /// ``this->size() + new_status_with_size.size()``
  ///
  /// The resulting status will be Ok if both statuses are ``Ok``,
  /// otherwise it will take on the earliest non-``Ok`` status.
  constexpr void UpdateAndAdd(StatusWithSize new_status_with_size) {
    Status new_status;
    if (ok()) {
      new_status = new_status_with_size.status();
    } else {
      new_status = status();
    }
    size_t new_size = size() + new_status_with_size.size();
    *this = StatusWithSize(new_status, new_size);
  }

  /// Zeros this size if the status is not ``Ok``.
  constexpr void ZeroIfNotOk() {
    if (!ok()) {
      *this = StatusWithSize(status(), 0);
    }
  }

  // Returns the size. The size is always present, even if status() is an error.
  [[nodiscard]] constexpr size_t size() const { return size_ & kSizeMask; }

  // Returns the size if the status() == OkStatus(), or the given default value
  // if status() is an error.
  [[nodiscard]] constexpr size_t size_or(size_t default_value) {
    return ok() ? size() : default_value;
  }

  // The maximum valid value for size.
  [[nodiscard]] static constexpr size_t max_size() { return kSizeMask; }

  // True if status() == OkStatus().
  [[nodiscard]] constexpr bool ok() const {
    return (size_ & kStatusMask) == 0u;
  }

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  constexpr void IgnoreError() const {}

  [[nodiscard]] constexpr Status status() const {
    return static_cast<Status::Code>((size_ & kStatusMask) >> kStatusShift);
  }

  // Functions for checking which status the StatusWithSize contains.
  [[nodiscard]] constexpr bool IsCancelled() const {
    return status().IsCancelled();
  }
  [[nodiscard]] constexpr bool IsUnknown() const {
    return status().IsUnknown();
  }
  [[nodiscard]] constexpr bool IsInvalidArgument() const {
    return status().IsInvalidArgument();
  }
  [[nodiscard]] constexpr bool IsDeadlineExceeded() const {
    return status().IsDeadlineExceeded();
  }
  [[nodiscard]] constexpr bool IsNotFound() const {
    return status().IsNotFound();
  }
  [[nodiscard]] constexpr bool IsAlreadyExists() const {
    return status().IsAlreadyExists();
  }
  [[nodiscard]] constexpr bool IsPermissionDenied() const {
    return status().IsPermissionDenied();
  }
  [[nodiscard]] constexpr bool IsResourceExhausted() const {
    return status().IsResourceExhausted();
  }
  [[nodiscard]] constexpr bool IsFailedPrecondition() const {
    return status().IsFailedPrecondition();
  }
  [[nodiscard]] constexpr bool IsAborted() const {
    return status().IsAborted();
  }
  [[nodiscard]] constexpr bool IsOutOfRange() const {
    return status().IsOutOfRange();
  }
  [[nodiscard]] constexpr bool IsUnimplemented() const {
    return status().IsUnimplemented();
  }
  [[nodiscard]] constexpr bool IsInternal() const {
    return status().IsInternal();
  }
  [[nodiscard]] constexpr bool IsUnavailable() const {
    return status().IsUnavailable();
  }
  [[nodiscard]] constexpr bool IsDataLoss() const {
    return status().IsDataLoss();
  }
  [[nodiscard]] constexpr bool IsUnauthenticated() const {
    return status().IsUnauthenticated();
  }

private:
  static constexpr size_t kStatusBits = 5;
  static constexpr size_t kSizeMask = ~static_cast<size_t>(0) >> kStatusBits;
  static constexpr size_t kStatusMask = ~kSizeMask;
  static constexpr size_t kStatusShift = sizeof(size_t) * 8 - kStatusBits;

  size_t size_;
};

[[nodiscard]] constexpr Status OkStatus() { return Status(); }

#endif // LLVM_LIBC_SRC_STDLIB_STATUS_H
