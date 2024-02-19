//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_ATOMIC_SYNC_H
#define _LIBCPP___ATOMIC_ATOMIC_SYNC_H

#include <__atomic/contention_t.h>
#include <__atomic/cxx_atomic_impl.h>
#include <__atomic/memory_order.h>
#include <__availability>
#include <__chrono/duration.h>
#include <__config>
#include <__memory/addressof.h>
#include <__thread/poll_with_backoff.h>
#include <__thread/support.h>
#include <__type_traits/conjunction.h>
#include <__type_traits/decay.h>
#include <__type_traits/invoke.h>
#include <__utility/declval.h>
#include <cstring>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// The customisation points to enable the following functions:
// - __atomic_wait
// - __atomic_wait_unless
// - __atomic_notify_one
// - __atomic_notify_all
// Note that std::atomic<T>::wait was back-ported to C++03
// there the below implementations look ugly to support C++03

// NOLINTBEGIN(libcpp-robust-against-adl)
struct __atomic_load_cpo {
  template <class _Tp>
  using _Ret = decltype(__tag_invoke(
      std::declval<__atomic_load_cpo>(), std::declval<const _Tp&>(), std::declval<memory_order>()));

  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI _Ret<_Tp> operator()(const _Tp& __t, memory_order __order) const _NOEXCEPT {
    return __tag_invoke(*this, __t, __order);
  }
};
// TODO: if we can deprecate std::atomic<T>::wait before c++17, we could add
// inline constexpr __atomic_load_cpo __atomic_load{};

struct __atomic_contention_address_cpo {
  template <class _Tp>
  using _Ret = decltype(__tag_invoke(std::declval<__atomic_contention_address_cpo>(), std::declval<const _Tp&>()));

  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI _Ret<_Tp> operator()(const _Tp& __t) const _NOEXCEPT {
    return __tag_invoke(*this, __t);
  }
};
// TODO: if we can deprecate std::atomic<T>::wait before c++17, we could add
// inline constexpr __atomic_contention_address_cpo __atomic_contention_address{};

// NOLINTEND(libcpp-robust-against-adl)

template <class _Tp>
using __atomic_waitable =
    _And<__invokable<__atomic_load_cpo, const _Tp&, memory_order>,
         __invokable<__atomic_contention_address_cpo, const _Tp&> >;

template <class _AtomicWaitable, class _Poll>
struct __atomic_wait_poll_impl {
  const _AtomicWaitable& __a_;
  _Poll __poll_;
  memory_order __order_;

  _LIBCPP_HIDE_FROM_ABI bool operator()() const {
    __atomic_load_cpo __atomic_load = {};
    auto __current_val              = __atomic_load(__a_, __order_);
    return __poll_(__current_val);
  }
};

#ifndef _LIBCPP_HAS_NO_THREADS

_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_one(void const volatile*);
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_all(void const volatile*);
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t __libcpp_atomic_monitor(void const volatile*);
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_wait(void const volatile*, __cxx_contention_t);

_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__cxx_atomic_notify_one(__cxx_atomic_contention_t const volatile*);
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__cxx_atomic_notify_all(__cxx_atomic_contention_t const volatile*);
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t
__libcpp_atomic_monitor(__cxx_atomic_contention_t const volatile*);
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait(__cxx_atomic_contention_t const volatile*, __cxx_contention_t);

template <class _AtomicWaitable, class _Poll>
struct __atomic_wait_backoff_impl {
  const _AtomicWaitable& __a_;
  _Poll __poll_;
  memory_order __order_;

  _LIBCPP_AVAILABILITY_SYNC
  _LIBCPP_HIDE_FROM_ABI bool
  __update_monitor_val_and_poll(__cxx_atomic_contention_t const volatile*, __cxx_contention_t& __monitor_val) const {
    // In case the contention type happens to be __cxx_atomic_contention_t, i.e. __cxx_atomic_impl<int64_t>,
    // the platform wait is directly monitoring the atomic value itself.
    __atomic_load_cpo __atomic_load = {};
    __monitor_val                   = __atomic_load(__a_, __order_);
    return __poll_(__monitor_val);
  }

  _LIBCPP_AVAILABILITY_SYNC
  _LIBCPP_HIDE_FROM_ABI bool
  __update_monitor_val_and_poll(void const volatile* __contention_address, __cxx_contention_t& __monitor_val) const {
    // In case the contention type is anything else, platform wait is monitoring a __cxx_atomic_contention_t
    // from the global pool, the monitor comes from __libcpp_atomic_monitor
    __monitor_val                   = std::__libcpp_atomic_monitor(__contention_address);
    __atomic_load_cpo __atomic_load = {};
    auto __current_val              = __atomic_load(__a_, __order_);
    return __poll_(__current_val);
  }

  _LIBCPP_AVAILABILITY_SYNC
  _LIBCPP_HIDE_FROM_ABI bool operator()(chrono::nanoseconds __elapsed) const {
    if (__elapsed > chrono::microseconds(64)) {
      __atomic_contention_address_cpo __atomic_contention_address = {};
      auto __contention_address                                   = __atomic_contention_address(__a_);
      __cxx_contention_t __monitor_val;
      if (__update_monitor_val_and_poll(__contention_address, __monitor_val))
        return true;
      std::__libcpp_atomic_wait(__contention_address, __monitor_val);
    } else if (__elapsed > chrono::microseconds(4))
      __libcpp_thread_yield();
    else {
    } // poll
    return false;
  }
};

template <class _AtomicWaitable, class _Poll>
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_HIDE_FROM_ABI void
__atomic_wait_unless(const _AtomicWaitable& __a, _Poll&& __poll, memory_order __order) {
  static_assert(__atomic_waitable<_AtomicWaitable>::value, "");
  __atomic_wait_poll_impl<_AtomicWaitable, __decay_t<_Poll> > __poll_impl     = {__a, __poll, __order};
  __atomic_wait_backoff_impl<_AtomicWaitable, __decay_t<_Poll> > __backoff_fn = {__a, __poll, __order};
  std::__libcpp_thread_poll_with_backoff(__poll_impl, __backoff_fn);
}

template <class _AtomicWaitable>
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_HIDE_FROM_ABI void __atomic_notify_one(const _AtomicWaitable& __a) {
  static_assert(__atomic_waitable<_AtomicWaitable>::value, "");
  __atomic_contention_address_cpo __atomic_contention_address = {};
  std::__cxx_atomic_notify_one(__atomic_contention_address(__a));
}

template <class _AtomicWaitable>
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_HIDE_FROM_ABI void __atomic_notify_all(const _AtomicWaitable& __a) {
  static_assert(__atomic_waitable<_AtomicWaitable>::value, "");
  __atomic_contention_address_cpo __atomic_contention_address = {};
  std::__cxx_atomic_notify_all(__atomic_contention_address(__a));
}

#else // _LIBCPP_HAS_NO_THREADS

template <class _AtomicWaitable, class _Poll>
_LIBCPP_HIDE_FROM_ABI void __atomic_wait_unless(const _AtomicWaitable& __a, _Poll&& __poll, memory_order __order) {
  __atomic_wait_poll_impl<_AtomicWaitable, __decay_t<_Poll> > __poll_fn = {__a, __poll, __order};
  std::__libcpp_thread_poll_with_backoff(__poll_fn, __spinning_backoff_policy());
}

template <class _AtomicWaitable>
_LIBCPP_HIDE_FROM_ABI void __atomic_notify_one(const _AtomicWaitable&) {}

template <class _AtomicWaitable>
_LIBCPP_HIDE_FROM_ABI void __atomic_notify_all(const _AtomicWaitable&) {}

#endif // _LIBCPP_HAS_NO_THREADS

template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI bool __cxx_nonatomic_compare_equal(_Tp const& __lhs, _Tp const& __rhs) {
  return std::memcmp(std::addressof(__lhs), std::addressof(__rhs), sizeof(_Tp)) == 0;
}

template <class _Tp>
struct __bind_nonatomic_equal {
  _Tp __val_;
  _LIBCPP_HIDE_FROM_ABI bool operator()(const _Tp& __arg) const {
    return !std::__cxx_nonatomic_compare_equal(__arg, __val_);
  }
};

template <class _AtomicWaitable, class _Up>
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_HIDE_FROM_ABI void
__atomic_wait(_AtomicWaitable& __a, _Up __val, memory_order __order) {
  __bind_nonatomic_equal<_Up> __nonatomic_equal = {__val};
  std::__atomic_wait_unless(__a, __nonatomic_equal, __order);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_ATOMIC_SYNC_H
