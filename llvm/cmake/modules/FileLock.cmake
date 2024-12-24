# CMake script that synchronizes process execution on a given file lock.
#
# Input variables:
#   LOCK_FILE_PATH    - The file to be locked for the scope of the process of this cmake script.
#   COMMAND           - The command to be executed.

file(LOCK ${LOCK_FILE_PATH})
separate_arguments(command_args NATIVE_COMMAND "${COMMAND}")
execute_process(COMMAND ${command_args} COMMAND_ERROR_IS_FATAL ANY)
