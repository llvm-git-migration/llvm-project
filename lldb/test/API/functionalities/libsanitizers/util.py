import subprocess

def no_libsanitizers():
    exe = shutil.which("dyld_shared_cache_util")
    if not exe:
        return "dyld_shared_cache_util not found"

    libs = subprocess.check_output([exe, "-list"]).decode("utf-8")
    for line in libs.split("\n"):
        if "libsystem_sanitizers.dylib" in line:
            return None

    return "libsystem_sanitizers.dylib not found in shared cache"
