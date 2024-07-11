add_lldb_library(lldbPluginArchitectureRISCV PLUGIN
  ArchitectureRISCV.cpp
  DirectCallReplacementPass.cpp

  LINK_LIBS
    lldbPluginProcessUtility
    lldbCore
    lldbTarget
    lldbUtility
  LINK_COMPONENTS
    Support
  )
