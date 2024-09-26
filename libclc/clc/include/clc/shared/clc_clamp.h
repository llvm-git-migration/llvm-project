#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible clamp
#define __clc_clamp clamp
#else

#define __CLC_BODY <clc/shared/clc_clamp.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <clc/shared/clc_clamp.inc>
#include <clc/math/gentype.inc>

#endif
