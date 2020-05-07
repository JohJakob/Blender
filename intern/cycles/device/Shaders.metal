//
//  Shaders.metal
//  bf_intern_cycles
//
//  Created by Andrew Fitzpatrick on 3/2/20.
//

#include <metal_stdlib>

#define __KERNEL_GPU__
#define __KERNEL_METAL__
#define CCL_NAMESPACE_BEGIN
#define CCL_NAMESPACE_END
#define ccl_device_inline inline
#define ccl_device_forceinline inline
#define ccl_device
#define ccl_device_noinline
#define ccl_global device
#define ccl_addr_space
#define ccl_align(x)
#define ccl_ref

#ifndef NULL
#define NULL (0)
#endif // null

using namespace metal;

#define make_int4 int4

inline int as_int(uint i) {
  return as_type<int>(i);
}

inline uint as_uint(int i) {
  return as_type<uint>(i);
}

inline uint as_uint(float f) {
  return as_type<uint>(f);
}

inline int __float_as_int(float f) {
  return as_type<int>(f);
}

inline float __int_as_float(int i) {
  return as_type<float>(i);
}

inline uint __float_as_uint(float f) {
  return as_type<uint>(f);
}

inline float __uint_as_float(uint i) {
  return as_type<float>(i);
}

#define make_float2 float2
#define make_float3 float3
#define make_float4 float4
#define make_int2 int2
#define make_int3 int3
#define make_int4 int4
#define make_uchar2 uchar2
#define make_uchar3 uchar3
#define make_uchar4 uchar4

#define floorf floor
#define fminf fmin
#define fmaxf fmax
#define fabsf fabs
#define ceilf ceil
//#define sqrtf sqrt
#define powf pow
#define expf exp
#define logf log
#define sinf sin
#define cosf cos
#define asinf asin
#define acosf acos
#define fmodf fmod
#define atan2f atan2
#define copysignf copysign

inline float sqrtf(float x) {
  return sqrt(x);
}

inline float3 rcp(thread const float3 &a) {
  return float3(1.0f) / a;
}

inline float lgammaf(float x) {
    // TODO
    return 0;
}

#define __device_space device

// todo make this work
#define __thread_space thread
#define ccl_private thread
#define ccl_constant constant

#define kernel_data (*kg->data)
#define kernel_assert
#define kernel_tex_array(tex) (kg->tex)

#include "util/util_transform.h"
#include "util/util_atomic.h"
#include "kernel/kernel_math.h"
#include "kernel/kernel_types.h"
#include "util/util_half.h"
#include "kernel/kernel_globals.h"
#include "kernel/kernel_color.h"

/* w0, w1, w2, and w3 are the four cubic B-spline basis functions. */
ccl_device float cubic_w0(float a)
{
  return (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
}

ccl_device float cubic_w1(float a)
{
  return (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
}

ccl_device float cubic_w2(float a)
{
  return (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
}

ccl_device float cubic_w3(float a)
{
  return (1.0f / 6.0f) * (a * a * a);
}

/* g0 and g1 are the two amplitude functions. */
ccl_device float cubic_g0(float a)
{
  return cubic_w0(a) + cubic_w1(a);
}

ccl_device float cubic_g1(float a)
{
  return cubic_w2(a) + cubic_w3(a);
}

/* h0 and h1 are the two offset functions */
ccl_device float cubic_h0(float a)
{
  /* Note +0.5 offset to compensate for CUDA linear filtering convention. */
  return -1.0f + cubic_w1(a) / (cubic_w0(a) + cubic_w1(a)) + 0.5f;
}

ccl_device float cubic_h1(float a)
{
  return 1.0f + cubic_w3(a) / (cubic_w2(a) + cubic_w3(a)) + 0.5f;
}

constexpr sampler s(coord::normalized,
                    address::repeat,
                    filter::linear);

/* Fast bicubic texture lookup using 4 bilinear lookups, adapted from CUDA samples. */
template<typename T, typename U>
ccl_device T
kernel_tex_image_interp_bicubic(constant const TextureInfo &info, texture2d<U> tex, float x, float y)
{
  x = (x * info.width) - 0.5f;
  y = (y * info.height) - 0.5f;

  float px = floor(x);
  float py = floor(y);
  float fx = x - px;
  float fy = y - py;

  float g0x = cubic_g0(fx);
  float g1x = cubic_g1(fx);
  float x0 = (px + cubic_h0(fx)) / info.width;
  float x1 = (px + cubic_h1(fx)) / info.width;
  float y0 = (py + cubic_h0(fy)) / info.height;
  float y1 = (py + cubic_h1(fy)) / info.height;

  return cubic_g0(fy) * (g0x * tex.sample(s, float2(x0, y0)) + g1x * tex.sample(s, float2(x1, y0))) +
         cubic_g1(fy) * (g0x * tex.sample(s, float2(x0, y1)) + g1x * tex.sample(s, float2(x1, y1)));
}

/* Fast tricubic texture lookup using 8 trilinear lookups. */
template<typename T, typename U>
ccl_device T kernel_tex_image_interp_bicubic_3d(
    constant const TextureInfo &info, texture3d<U> tex, float x, float y, float z)
{
  x = (x * info.width) - 0.5f;
  y = (y * info.height) - 0.5f;
  z = (z * info.depth) - 0.5f;

  float px = floor(x);
  float py = floor(y);
  float pz = floor(z);
  float fx = x - px;
  float fy = y - py;
  float fz = z - pz;

  float g0x = cubic_g0(fx);
  float g1x = cubic_g1(fx);
  float g0y = cubic_g0(fy);
  float g1y = cubic_g1(fy);
  float g0z = cubic_g0(fz);
  float g1z = cubic_g1(fz);

  float x0 = (px + cubic_h0(fx)) / info.width;
  float x1 = (px + cubic_h1(fx)) / info.width;
  float y0 = (py + cubic_h0(fy)) / info.height;
  float y1 = (py + cubic_h1(fy)) / info.height;
  float z0 = (pz + cubic_h0(fz)) / info.depth;
  float z1 = (pz + cubic_h1(fz)) / info.depth;

  return g0z * (g0y * (g0x * tex.sample(s, float3(x0, y0, z0)) + g1x * tex.sample(s, float3(x1, y0, z0))) +
                g1y * (g0x * tex.sample(s, float3(x0, y1, z0)) + g1x * tex.sample(s, float3(x1, y1, z0)))) +
         g1z * (g0y * (g0x * tex.sample(s, float3(x0, y0, z1)) + g1x * tex.sample(s, float3(x1, y0, z1))) +
                g1y * (g0x * tex.sample(s, float3(x0, y1, z1)) + g1x * tex.sample(s, float3(x1, y1, z1))));
}



ccl_device float4 kernel_tex_image_interp_3d(thread KernelGlobals *kg,
                                             int id,
                                             float3 P,
                                             InterpolationType interp)
{
    const constant TextureInfo &info = kg->textureInfo[id];

    if (info.use_transform_3d) {
        Transform tf = info.transform_3d;
        P = transform_point(&tf, P);
    }

    const float x = P.x;
    const float y = P.y;
    const float z = P.z;

    constexpr sampler s(coord::pixel,
                        address::clamp_to_zero,
                        filter::nearest);

    const constant texture3d<float> &tex = *info.data3d;
    uint interpolation = (interp == INTERPOLATION_NONE) ? info.interpolation : interp;

    const int texture_type = info.data_type;
    if (texture_type == IMAGE_DATA_TYPE_FLOAT4 || texture_type == IMAGE_DATA_TYPE_BYTE4 ||
        texture_type == IMAGE_DATA_TYPE_HALF4 || texture_type == IMAGE_DATA_TYPE_USHORT4) {
      if (interpolation == INTERPOLATION_CUBIC) {
        return kernel_tex_image_interp_bicubic_3d<float4, float>(info, tex, x, y, z);
      }
      else {
          return tex.sample(s, P);
      }
    }
    else {
      float f;

      if (interpolation == INTERPOLATION_CUBIC) {
        f = kernel_tex_image_interp_bicubic_3d<float, float>(info, tex, x, y, z);
      }
      else {
          f = tex.sample(s, P).x;
      }

      return make_float4(f, f, f, 1.0f);
    }

}

ccl_device float4 kernel_tex_image_interp(thread KernelGlobals *kg, int id, float x, float y) {

    const constant TextureInfo &info = kg->textureInfo[id];
    const constant texture2d<float> &tex = *info.data2d;

    float2 P = float2(x, y);

    /* float4, byte4, ushort4 and half4 */
    const int texture_type = info.data_type;
    if (texture_type == IMAGE_DATA_TYPE_FLOAT4 || texture_type == IMAGE_DATA_TYPE_BYTE4 ||
        texture_type == IMAGE_DATA_TYPE_HALF4 || texture_type == IMAGE_DATA_TYPE_USHORT4) {
        if (info.interpolation == INTERPOLATION_CUBIC) {
            return kernel_tex_image_interp_bicubic<float4, float>(info, tex, x, y);
        }
        else {
            return tex.sample(s, P);
        }
    }
    /* float, byte and half */
    else {
        float f;

        if (info.interpolation == INTERPOLATION_CUBIC) {
            f = kernel_tex_image_interp_bicubic<float, float>(info, tex, x, y);
        }
        else {
            f = tex.sample(s, P).x;
        }

        return make_float4(f, f, f, 1.0f);
    }
}

//
//#include "kernel/kernel_film.h"
//#include "kernel/kernel_path.h"
//#include "kernel/kernel_path_branched.h"
//#include "kernel/kernel_bake.h"
//#include "kernel/kernel_work_stealing.h"
//
//kernel void
//path_trace(device WorkTile* tile,
//           device uint *total_work_size,
//           uint2 grid_size [[grid_size]],
//           uint2 thread_position [[thread_position_in_grid]]) {
//  uint thread_index = thread_position.y * grid_size.x + thread_position.x;
//
//  bool thread_is_active = thread_index < *total_work_size;
//
//  uint x, y, sample;
//  KernelGlobals kg;
//  if (thread_is_active) {
//    get_work_pixel(tile, work_index, &x, &y, &sample);
//
//
//  }
//}

#define ccl_device_noinline_cpu
#define ccl_static_constant constant
#define tanf tan
#define atanf atan
#define sinhf sinh
#define coshf cosh
#define tanhf tanh

#define ATTR_FALLTHROUGH

#include "kernel/kernel_projection.h"
#include "kernel/geom/geom_triangle.h"
#include "kernel/geom/geom_object.h"
#include "kernel/geom/geom_attribute.h"
#include "kernel/geom/geom_volume.h"
#include "kernel/geom/geom_motion_curve.h"
#include "kernel/geom/geom_curve.h"
#include "kernel/geom/geom_patch.h"
#include "kernel/geom/geom_subd_triangle.h"
#include "kernel/geom/geom_primitive.h"
#include "kernel/kernel_write_passes.h"
#include "kernel/kernel_accumulate.h"
#include "kernel/kernel_random.h"
//#include "kernel/kernel_jitter.h"
#include "kernel/kernel_montecarlo.h"
#include "kernel/kernel_shader.h"
#include "kernel/kernel_volume.h"
#include "kernel/kernel_path_state.h"
#include "kernel/kernel_bake.h"

kernel void kernel_metal_background(device uint4 *input,
                                    device float4 *output,
                                    device int *type,
                                    device int *filterType,
                                    uint2 grid_size [[grid_size]],
                                    uint2 thread_position [[thread_position_in_grid]]) {
    uint x = thread_position.x;

    if (x < grid_size.x) {
        // do stuff
        KernelGlobals kg;
        kernel_bake_evaluate(&kg, input, output, (ShaderEvalType)*type, *filterType, x, 0, 0);
    }
}
