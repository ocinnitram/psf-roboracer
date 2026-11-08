// Correctly rounded atan from CORE-MATH
// (https://gitlab.inria.fr/core-math/core-math, MIT, commit 6b844573),
// vendored unchanged beside this header.
//
// The model calls cr_atan instead of std::atan: glibc's atan differs in the
// last bit between releases (2.35 and 2.44), and one bit in the tyre force is
// enough for two machines to certify the same layout differently. A correctly
// rounded result is unique, so it cannot depend on the libm.
#ifndef SF_CORE_MATH_H
#define SF_CORE_MATH_H

extern "C" double cr_atan(double x);

#endif
