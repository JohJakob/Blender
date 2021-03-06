/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

CCL_NAMESPACE_BEGIN

/* Curve Primitive
 *
 * Curve primitive for rendering hair and fur. These can be render as flat
 * ribbons or curves with actual thickness. The curve can also be rendered as
 * line segments rather than curves for better performance.
 */

#ifdef __HAIR__

/* Interpolation of curve geometry */

ccl_device_inline float3 curvetangent(float t, float3 p0, float3 p1, float3 p2, float3 p3)
{
  float fc = 0.71f;
  float data[4];
  float t2 = t * t;
  data[0] = -3.0f * fc * t2 + 4.0f * fc * t - fc;
  data[1] = 3.0f * (2.0f - fc) * t2 + 2.0f * (fc - 3.0f) * t;
  data[2] = 3.0f * (fc - 2.0f) * t2 + 2.0f * (3.0f - 2.0f * fc) * t + fc;
  data[3] = 3.0f * fc * t2 - 2.0f * fc * t;
  return data[0] * p0 + data[1] * p1 + data[2] * p2 + data[3] * p3;
}

ccl_device_inline float3 curvepoint(float t, float3 p0, float3 p1, float3 p2, float3 p3)
{
  float data[4];
  float fc = 0.71f;
  float t2 = t * t;
  float t3 = t2 * t;
  data[0] = -fc * t3 + 2.0f * fc * t2 - fc * t;
  data[1] = (2.0f - fc) * t3 + (fc - 3.0f) * t2 + 1.0f;
  data[2] = (fc - 2.0f) * t3 + (3.0f - 2.0f * fc) * t2 + fc * t;
  data[3] = fc * t3 - fc * t2;
  return data[0] * p0 + data[1] * p1 + data[2] * p2 + data[3] * p3;
}

/* Reading attributes on various curve elements */

ccl_device float curve_attribute_float(__device_space KernelGlobals *kg,
                                       __device_space const ShaderData *sd,
                                       const AttributeDescriptor desc,
                                       __thread_space float *dx,
                                       __thread_space float *dy)
{
  if (desc.element == ATTR_ELEMENT_CURVE) {
#  ifdef __RAY_DIFFERENTIALS__
    if (dx)
      *dx = 0.0f;
    if (dy)
      *dy = 0.0f;
#  endif

    return kernel_tex_fetch(__attributes_float, desc.offset + sd->prim);
  }
  else if (desc.element == ATTR_ELEMENT_CURVE_KEY ||
           desc.element == ATTR_ELEMENT_CURVE_KEY_MOTION) {
    float4 curvedata = kernel_tex_fetch(__curves, sd->prim);
    int k0 = __float_as_int(curvedata.x) + PRIMITIVE_UNPACK_SEGMENT(sd->type);
    int k1 = k0 + 1;

    float f0 = kernel_tex_fetch(__attributes_float, desc.offset + k0);
    float f1 = kernel_tex_fetch(__attributes_float, desc.offset + k1);

#  ifdef __RAY_DIFFERENTIALS__
    if (dx)
      *dx = sd->du.dx * (f1 - f0);
    if (dy)
      *dy = 0.0f;
#  endif

    return (1.0f - sd->u) * f0 + sd->u * f1;
  }
  else if (desc.element == ATTR_ELEMENT_OBJECT || desc.element == ATTR_ELEMENT_MESH) {
#  ifdef __RAY_DIFFERENTIALS__
    if (dx)
      *dx = 0.0f;
    if (dy)
      *dy = 0.0f;
#  endif

    return kernel_tex_fetch(__attributes_float, desc.offset);
  }
  else {
#  ifdef __RAY_DIFFERENTIALS__
    if (dx)
      *dx = 0.0f;
    if (dy)
      *dy = 0.0f;
#  endif

    return 0.0f;
  }
}

ccl_device float2 curve_attribute_float2(__device_space KernelGlobals *kg,
                                         __device_space const ShaderData *sd,
                                         const AttributeDescriptor desc,
                                         __thread_space float2 *dx,
                                         __thread_space float2 *dy)
{
  if (desc.element == ATTR_ELEMENT_CURVE) {
    /* idea: we can't derive any useful differentials here, but for tiled
     * mipmap image caching it would be useful to avoid reading the highest
     * detail level always. maybe a derivative based on the hair density
     * could be computed somehow? */
#  ifdef __RAY_DIFFERENTIALS__
    if (dx)
      *dx = make_float2(0.0f, 0.0f);
    if (dy)
      *dy = make_float2(0.0f, 0.0f);
#  endif

    return kernel_tex_fetch(__attributes_float2, desc.offset + sd->prim);
  }
  else if (desc.element == ATTR_ELEMENT_CURVE_KEY ||
           desc.element == ATTR_ELEMENT_CURVE_KEY_MOTION) {
    float4 curvedata = kernel_tex_fetch(__curves, sd->prim);
    int k0 = __float_as_int(curvedata.x) + PRIMITIVE_UNPACK_SEGMENT(sd->type);
    int k1 = k0 + 1;

    float2 f0 = kernel_tex_fetch(__attributes_float2, desc.offset + k0);
    float2 f1 = kernel_tex_fetch(__attributes_float2, desc.offset + k1);

#  ifdef __RAY_DIFFERENTIALS__
    if (dx)
      *dx = sd->du.dx * (f1 - f0);
    if (dy)
      *dy = make_float2(0.0f, 0.0f);
#  endif

    return (1.0f - sd->u) * f0 + sd->u * f1;
  }
  else if (desc.element == ATTR_ELEMENT_OBJECT || desc.element == ATTR_ELEMENT_MESH) {
#  ifdef __RAY_DIFFERENTIALS__
    if (dx)
      *dx = make_float2(0.0f, 0.0f);
    if (dy)
      *dy = make_float2(0.0f, 0.0f);
#  endif

    return kernel_tex_fetch(__attributes_float2, desc.offset);
  }
  else {
#  ifdef __RAY_DIFFERENTIALS__
    if (dx)
      *dx = make_float2(0.0f, 0.0f);
    if (dy)
      *dy = make_float2(0.0f, 0.0f);
#  endif

    return make_float2(0.0f, 0.0f);
  }
}

ccl_device float3 curve_attribute_float3(__device_space KernelGlobals *kg,
                                         __device_space const ShaderData *sd,
                                         const AttributeDescriptor desc,
                                         __thread_space float3 *dx,
                                         __thread_space float3 *dy)
{
  if (desc.element == ATTR_ELEMENT_CURVE) {
    /* idea: we can't derive any useful differentials here, but for tiled
     * mipmap image caching it would be useful to avoid reading the highest
     * detail level always. maybe a derivative based on the hair density
     * could be computed somehow? */
#  ifdef __RAY_DIFFERENTIALS__
    if (dx)
      *dx = make_float3(0.0f, 0.0f, 0.0f);
    if (dy)
      *dy = make_float3(0.0f, 0.0f, 0.0f);
#  endif

    return float4_to_float3(kernel_tex_fetch(__attributes_float3, desc.offset + sd->prim));
  }
  else if (desc.element == ATTR_ELEMENT_CURVE_KEY ||
           desc.element == ATTR_ELEMENT_CURVE_KEY_MOTION) {
    float4 curvedata = kernel_tex_fetch(__curves, sd->prim);
    int k0 = __float_as_int(curvedata.x) + PRIMITIVE_UNPACK_SEGMENT(sd->type);
    int k1 = k0 + 1;

    float3 f0 = float4_to_float3(kernel_tex_fetch(__attributes_float3, desc.offset + k0));
    float3 f1 = float4_to_float3(kernel_tex_fetch(__attributes_float3, desc.offset + k1));

#  ifdef __RAY_DIFFERENTIALS__
    if (dx)
      *dx = sd->du.dx * (f1 - f0);
    if (dy)
      *dy = make_float3(0.0f, 0.0f, 0.0f);
#  endif

    return (1.0f - sd->u) * f0 + sd->u * f1;
  }
  else if (desc.element == ATTR_ELEMENT_OBJECT || desc.element == ATTR_ELEMENT_MESH) {
#  ifdef __RAY_DIFFERENTIALS__
    if (dx)
      *dx = make_float3(0.0f, 0.0f, 0.0f);
    if (dy)
      *dy = make_float3(0.0f, 0.0f, 0.0f);
#  endif

    return float4_to_float3(kernel_tex_fetch(__attributes_float3, desc.offset));
  }
  else {
#  ifdef __RAY_DIFFERENTIALS__
    if (dx)
      *dx = make_float3(0.0f, 0.0f, 0.0f);
    if (dy)
      *dy = make_float3(0.0f, 0.0f, 0.0f);
#  endif

    return make_float3(0.0f, 0.0f, 0.0f);
  }
}

/* Curve thickness */

ccl_device float curve_thickness(__device_space KernelGlobals *kg, __device_space ShaderData *sd)
{
  float r = 0.0f;

  if (sd->type & PRIMITIVE_ALL_CURVE) {
    float4 curvedata = kernel_tex_fetch(__curves, sd->prim);
    int k0 = __float_as_int(curvedata.x) + PRIMITIVE_UNPACK_SEGMENT(sd->type);
    int k1 = k0 + 1;

    float4 P_curve[2];

    if (sd->type & PRIMITIVE_CURVE) {
      P_curve[0] = kernel_tex_fetch(__curve_keys, k0);
      P_curve[1] = kernel_tex_fetch(__curve_keys, k1);
    }
    else {
      motion_curve_keys(kg, sd->object, sd->prim, sd->time, k0, k1, P_curve);
    }

    r = (P_curve[1].w - P_curve[0].w) * sd->u + P_curve[0].w;
  }

  return r * 2.0f;
}

/* Curve location for motion pass, linear interpolation between keys and
 * ignoring radius because we do the same for the motion keys */

ccl_device float3 curve_motion_center_location(__device_space KernelGlobals *kg, __device_space ShaderData *sd)
{
  float4 curvedata = kernel_tex_fetch(__curves, sd->prim);
  int k0 = __float_as_int(curvedata.x) + PRIMITIVE_UNPACK_SEGMENT(sd->type);
  int k1 = k0 + 1;

  float4 P_curve[2];

  P_curve[0] = kernel_tex_fetch(__curve_keys, k0);
  P_curve[1] = kernel_tex_fetch(__curve_keys, k1);

  return float4_to_float3(P_curve[1]) * sd->u + float4_to_float3(P_curve[0]) * (1.0f - sd->u);
}

/* Curve tangent normal */

ccl_device float3 curve_tangent_normal(__device_space KernelGlobals *kg, __device_space ShaderData *sd)
{
  float3 tgN = make_float3(0.0f, 0.0f, 0.0f);

  if (sd->type & PRIMITIVE_ALL_CURVE) {

    tgN = -(-sd->I - sd->dPdu * (dot(sd->dPdu, -sd->I) / len_squared(sd->dPdu)));
    tgN = normalize(tgN);

    /* need to find suitable scaled gd for corrected normal */
#  if 0
    tgN = normalize(tgN - gd * sd->dPdu);
#  endif
  }

  return tgN;
}

/* Curve bounds utility function */

ccl_device_inline void curvebounds(__thread_space float *lower,
                                   __thread_space float *upper,
                                   __thread_space float *extremta,
                                   __thread_space float *extrema,
                                   __thread_space float *extremtb,
                                   __thread_space float *extremb,
                                   float p0,
                                   float p1,
                                   float p2,
                                   float p3)
{
  float halfdiscroot = (p2 * p2 - 3 * p3 * p1);
  float ta = -1.0f;
  float tb = -1.0f;

  *extremta = -1.0f;
  *extremtb = -1.0f;
  *upper = p0;
  *lower = (p0 + p1) + (p2 + p3);
  *extrema = *upper;
  *extremb = *lower;

  if (*lower >= *upper) {
    *upper = *lower;
    *lower = p0;
  }

  if (halfdiscroot >= 0) {
    float inv3p3 = (1.0f / 3.0f) / p3;
    halfdiscroot = sqrtf(halfdiscroot);
    ta = (-p2 - halfdiscroot) * inv3p3;
    tb = (-p2 + halfdiscroot) * inv3p3;
  }

  float t2;
  float t3;

  if (ta > 0.0f && ta < 1.0f) {
    t2 = ta * ta;
    t3 = t2 * ta;
    *extremta = ta;
    *extrema = p3 * t3 + p2 * t2 + p1 * ta + p0;

    *upper = fmaxf(*extrema, *upper);
    *lower = fminf(*extrema, *lower);
  }

  if (tb > 0.0f && tb < 1.0f) {
    t2 = tb * tb;
    t3 = t2 * tb;
    *extremtb = tb;
    *extremb = p3 * t3 + p2 * t2 + p1 * tb + p0;

    *upper = fmaxf(*extremb, *upper);
    *lower = fminf(*extremb, *lower);
  }
}

#endif /* __HAIR__ */

CCL_NAMESPACE_END
