def no_libsanitizers(testbase):
    testbase.runCmd("image list")
    return not "libsystem_sanitizers.dylib" in testbase.res.GetOutput()
