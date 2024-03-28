add_entrypoint_object(
  epoll_wait
  SRCS
    epoll_wait.cpp
  HDRS
    # TODO: these 2 should not be necessary
    ../../../signal/sigset_t.h
    ../../../time/struct_timespec.h

    ../epoll_wait.h
  DEPENDS
    libc.include.sys_syscall
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
    libc.src.sys.epoll.struct_epoll_event
    # TODO: why don't these work for overlay mode?
    # libc.src.signal.sigset_t
    # libc.src.time.struct_timespec
)

add_entrypoint_object(
  epoll_pwait
  SRCS
    epoll_pwait.cpp
  HDRS
    # TODO: these 2 should not be necessary
    ../../../signal/sigset_t.h
    ../../../time/struct_timespec.h

    ../epoll_pwait.h
  DEPENDS
    libc.include.signal
    libc.include.sys_syscall
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
    libc.src.sys.epoll.struct_epoll_event
    # TODO: why don't these work for overlay mode?
    # libc.src.signal.sigset_t
    # libc.src.time.struct_timespec
)

add_entrypoint_object(
  epoll_pwait2
  SRCS
    epoll_pwait2.cpp
  HDRS
    # TODO: these 2 should not be necessary
    ../../../signal/sigset_t.h
    ../../../time/struct_timespec.h

    ../epoll_pwait2.h
  DEPENDS
    libc.include.signal
    libc.include.sys_syscall
    libc.include.time
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
    libc.src.sys.epoll.struct_epoll_event
    # TODO: why don't these work for overlay mode?
    # libc.src.signal.sigset_t
    # libc.src.time.struct_timespec
)
