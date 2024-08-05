add_entrypoint_object(
  raise
  SRCS
    raise.cpp
  HDRS
    ../raise.h
  DEPENDS
    libc.hdr.types.sigset_t
    libc.src.__support.RPC.rpc_client
)

