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
#include "clang/Tooling/ModuleBuildDaemon/SocketSupport.h"

using namespace clang;
using namespace llvm;

namespace cc1modbuildd {

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

template <typename T> std::string getBufferFromSocketMsg(T Msg) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value,
                "T must inherit from cc1modbuildd::BaseMsg");

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  llvm::yaml::Output YamlOut(OS);

  YamlOut << Msg;
  return Buffer;
}

template <typename T> Expected<T> getSocketMsgFromBuffer(StringRef Buffer) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value,
                "T must inherit from cc1modbuildd::BaseMsg");

  T ClientRequest;
  llvm::yaml::Input YamlIn(Buffer);
  YamlIn >> ClientRequest;

  // YamlIn.error() dumps an error message if there is one
  if (YamlIn.error()) {
    std::string Msg = "Syntax or semantic error during YAML parsing";
    return llvm::make_error<StringError>(Msg, inconvertibleErrorCode());
  }

  return ClientRequest;
}

template <typename T> Expected<T> readSocketMsgFromSocket(int FD) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value,
                "T must inherit from cc1modbuildd::BaseMsg");

  std::string BufferConsumer;
  if (llvm::Error ReadErr = readFromSocket(FD, BufferConsumer))
    return std::move(ReadErr);

  // Wait for response from module build daemon
  Expected<T> MaybeResponse =
      getSocketMsgFromBuffer<T>(std::move(BufferConsumer).c_str());
  if (!MaybeResponse)
    return std::move(MaybeResponse.takeError());
  return std::move(*MaybeResponse);
}

template <typename T> llvm::Error writeSocketMsgToSocket(T Msg, int FD) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value,
                "T must inherit from cc1modbuildd::BaseMsg");

  std::string Buffer = getBufferFromSocketMsg(Msg);
  if (llvm::Error Err = writeToSocket(Buffer, FD))
    return std::move(Err);

  return llvm::Error::success();
}

template <typename T>
Expected<int> connectAndWriteSocketMsgToSocket(T Msg, StringRef SocketPath) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value,
                "T must inherit from cc1modbuildd::BaseMsg");

  Expected<int> MaybeFD = connectToSocket(SocketPath);
  if (!MaybeFD)
    return std::move(MaybeFD.takeError());
  int FD = std::move(*MaybeFD);

  if (llvm::Error Err = writeSocketMsgToSocket(Msg, FD))
    return std::move(Err);

  return FD;
}

} // namespace cc1modbuildd

template <>
struct llvm::yaml::ScalarEnumerationTraits<cc1modbuildd::StatusType> {
  static void enumeration(IO &Io, cc1modbuildd::StatusType &Value) {
    Io.enumCase(Value, "REQUEST", cc1modbuildd::StatusType::REQUEST);
    Io.enumCase(Value, "SUCCESS", cc1modbuildd::StatusType::SUCCESS);
    Io.enumCase(Value, "FAILURE", cc1modbuildd::StatusType::FAILURE);
  }
};

template <>
struct llvm::yaml::ScalarEnumerationTraits<cc1modbuildd::ActionType> {
  static void enumeration(IO &Io, cc1modbuildd::ActionType &Value) {
    Io.enumCase(Value, "HANDSHAKE", cc1modbuildd::ActionType::HANDSHAKE);
  }
};

template <> struct llvm::yaml::MappingTraits<cc1modbuildd::HandshakeMsg> {
  static void mapping(IO &Io, cc1modbuildd::HandshakeMsg &Info) {
    Io.mapRequired("Action", Info.MsgAction);
    Io.mapRequired("Status", Info.MsgStatus);
  }
};

#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETMSGSUPPORT_H
