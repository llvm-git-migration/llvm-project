def RPCPortT : NamedType<"rpc_port_t">;
def RPCPortPtrT : PtrType<RPCPortT>;
def RPCBufferT : NamedType<"rpc_buffer_t">;
def RPCBufferPtrT : PtrType<RPCBufferT>;
def RPCCallbackT : NamedType<"rpc_callback_t">;

def GPUExtensions : StandardSpec<"GPUExtensions"> {
  HeaderSpec RPC = HeaderSpec<
    "gpu/rpc.h",
    [], // Macros
    [RPCPortT, RPCBufferT,  RPCCallbackT, SizeTType], // Types
    [], // Enumerations
    [
        FunctionSpec<
            "rpc_host_call",
            RetValSpec<VoidType>,
            [ArgSpec<VoidPtr>, ArgSpec<VoidPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
            "rpc_open_port",
            RetValSpec<RPCPortT>,
            [ArgSpec<UnsignedIntType>]
        >,
        FunctionSpec<
            "rpc_send",
            RetValSpec<VoidType>,
            [ArgSpec<RPCPortPtrT>, ArgSpec<RPCCallbackT>, ArgSpec<VoidPtr>]
        >,
        FunctionSpec<
            "rpc_recv",
            RetValSpec<VoidType>,
            [ArgSpec<RPCPortPtrT>, ArgSpec<RPCCallbackT>, ArgSpec<VoidPtr>]
        >,
        FunctionSpec<
            "rpc_close_port",
            RetValSpec<VoidType>,
            [ArgSpec<RPCPortPtrT>]
        >,
    ]
  >;
  let Headers = [
    RPC,
  ];
}
