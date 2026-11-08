/* Correctly rounded float functions from CORE-MATH
 * (https://gitlab.inria.fr/core-math/core-math, MIT, commit 6b844573),
 * vendored unchanged beside this header.
 *
 * The plant calls these instead of libm's tanf, atanf, asinf and atan2f.
 * glibc changed those four between releases (2.35 and 2.44 differ in the
 * last bit), and one bit in the tyre force is enough for two machines to
 * drive the same layout differently. A correctly rounded result is unique,
 * so it cannot depend on the libm; on 2 million inputs each it equals
 * glibc 2.44's output bit for bit. */
#ifndef CORE_MATH_H
#define CORE_MATH_H

float cr_tanf(float x);
float cr_atanf(float x);
float cr_asinf(float x);
float cr_atan2f(float y, float x);

#endif
