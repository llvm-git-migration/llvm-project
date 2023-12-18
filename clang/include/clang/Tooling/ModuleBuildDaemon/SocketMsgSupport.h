//===------------------------- SocketMsgSupport.h -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETMSGSUPPORT_H
#define LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETMSGSUPPORT_H

#include "clang/Tooling/ModuleBuildDaemon/Client.h"

namespace clang::tooling::cc1modbuildd {

enum class ActionType { HANDSHAKE };
enum class StatusType { REQUEST, SUCCESS, FAILURE };

struct BaseMsg {
  ActionType MsgAction;
  StatusType MsgStatus;

  BaseMsg() = default;
  BaseMsg(ActionType Action, StatusType Status)
      : MsgAction(Action), MsgStatus(Status) {}
};

struct HandshakeMsg : public BaseMsg {
  HandshakeMsg() = default;
  HandshakeMsg(ActionType Action, StatusType Status)
      : BaseMsg(Action, Status) {}
};

Expected<std::unique_ptr<llvm::raw_socket_stream>>
connectToSocket(StringRef SocketPath);
llvm::Error readFromSocket(llvm::raw_socket_stream &Connection,
                           std::string &BufferConsumer);
void writeToSocket(llvm::raw_socket_stream &Socket, llvm::StringRef Buffer);

template <typename T> std::string getBufferFromSocketMsg(T Msg) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value,
                "T must inherit from cc1modbuildd::BaseMsg");

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  llvm::yaml::Output YamlOut(OS);

  YamlOut << Msg;
  return Buffer;
}

template <typename T>
llvm::Expected<T> getSocketMsgFromBuffer(llvm::StringRef Buffer) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value,
                "T must inherit from cc1modbuildd::BaseMsg");

  T Request;
  llvm::yaml::Input YamlIn(Buffer);
  YamlIn >> Request;

  // YamlIn.error() dumps an error message if there is one
  if (YamlIn.error()) {
    std::string Msg = "Syntax or semantic error during YAML parsing";
    return llvm::make_error<llvm::StringError>(Msg,
                                               llvm::inconvertibleErrorCode());
  }

  return Request;
}

template <typename T>
llvm::Expected<T> readSocketMsgFromSocket(llvm::raw_socket_stream &Socket) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value,
                "T must inherit from cc1modbuildd::BaseMsg");

  std::string BufferConsumer;
  if (llvm::Error ReadErr = readFromSocket(Socket, BufferConsumer))
    return std::move(ReadErr);

  // Wait for response from module build daemon
  llvm::Expected<T> MaybeResponse =
      getSocketMsgFromBuffer<T>(std::move(BufferConsumer).c_str());
  if (!MaybeResponse)
    return std::move(MaybeResponse.takeError());
  return std::move(*MaybeResponse);
}

template <typename T>
llvm::Error writeSocketMsgToSocket(llvm::raw_socket_stream &Socket, T Msg) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value,
                "T must inherit from cc1modbuildd::BaseMsg");

  std::string Buffer = getBufferFromSocketMsg(Msg);
  writeToSocket(Socket, Buffer);

  return llvm::Error::success();
}

template <typename T>
llvm::Expected<int>
connectAndWriteSocketMsgToSocket(T Msg, llvm::StringRef SocketPath) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value,
                "T must inherit from cc1modbuildd::BaseMsg");

  llvm::Expected<int> MaybeFD = connectToSocket(SocketPath);
  if (!MaybeFD)
    return std::move(MaybeFD.takeError());
  int FD = std::move(*MaybeFD);

  if (llvm::Error Err = writeSocketMsgToSocket(Msg, FD))
    return std::move(Err);

  return FD;
}

} // namespace clang::tooling::cc1modbuildd

namespace cc1mod = clang::tooling::cc1modbuildd;

template <> struct llvm::yaml::ScalarEnumerationTraits<cc1mod::StatusType> {
  static void enumeration(IO &Io, cc1mod::StatusType &Value) {
    Io.enumCase(Value, "REQUEST", cc1mod::StatusType::REQUEST);
    Io.enumCase(Value, "SUCCESS", cc1mod::StatusType::SUCCESS);
    Io.enumCase(Value, "FAILURE", cc1mod::StatusType::FAILURE);
  }
};

template <> struct llvm::yaml::ScalarEnumerationTraits<cc1mod::ActionType> {
  static void enumeration(IO &Io, cc1mod::ActionType &Value) {
    Io.enumCase(Value, "HANDSHAKE", cc1mod::ActionType::HANDSHAKE);
  }
};

template <> struct llvm::yaml::MappingTraits<cc1mod::HandshakeMsg> {
  static void mapping(IO &Io, cc1mod::HandshakeMsg &Info) {
    Io.mapRequired("Action", Info.MsgAction);
    Io.mapRequired("Status", Info.MsgStatus);
  }
};

#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETMSGSUPPORT_H
