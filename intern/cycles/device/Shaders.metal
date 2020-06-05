//
//  Shaders.metal
//  bf_intern_cycles
//
//  Created by Andrew Fitzpatrick on 3/2/20.
//

#include <metal_stdlib>
#include <metal_atomic>

#define __KERNEL_GPU__
#define __KERNEL_METAL__
#define CCL_NAMESPACE_BEGIN
#define CCL_NAMESPACE_END
#define ccl_device_inline inline
#define ccl_device_forceinline inline
#define ccl_device
#define ccl_device_noinline
#define ccl_global
#define ccl_addr_space
#define ccl_align(x)
#define ccl_ref
#define ccl_local_param
#define ccl_local_id(x) 0
#define ccl_local_size(x) 0

#define ccl_global_id(x) (kg->global_id[x])
#define ccl_global_size(x) (kg->global_size[x])

#define ccl_barrier(x)
#define CCL_LOCAL_MEM_FENCE

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
#define tanf tan
#define atanf atan
#define sinhf sinh
#define coshf cosh
#define tanhf tanh


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
#define __thread_space thread
#define ccl_private thread
#define ccl_constant constant

#define kernel_data (*kg->data)
#define kernel_assert(x)
#define kernel_tex_array(tex) (kg->tex)

#define ccl_device_noinline_cpu
#define ccl_static_constant constant

#define ATTR_FALLTHROUGH
#define ccl_loop_no_unroll
#define ccl_optional_struct_init


//#define atomic_cmpxchg(A, B, C) atomic_exchange_explicit(A, C, memory_order_relaxed)

inline float atomic_add_and_fetch_float(volatile device float* source, const float operand) {
    auto intSource = (volatile device atomic_int *)source;
    int sourceInt = 0;
    int desiredInt = 0;
    do {
        sourceInt = atomic_load_explicit(intSource, memory_order_relaxed);
        float sourceFloat = as_type<float>(sourceInt);
        float desiredFloat = sourceFloat + operand;
        desiredInt = as_type<int>(desiredFloat);
    } while (atomic_compare_exchange_weak_explicit(intSource,
                                                   &sourceInt,
                                                   desiredInt,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed));
    return desiredInt;
}

inline float atomic_compare_and_swap_float(volatile device float *dest,
                                                      const float old_val,
                                                      const float new_val)
{
    auto intDest = (volatile device atomic_int *)dest;
    int oldInt = atomic_exchange_explicit(intDest, as_type<int>(new_val), memory_order_relaxed);
    return as_type<float>(oldInt);
}

inline uint atomic_fetch_and_or_uint32(device uint* input, uint M) {
    auto atomic_input = (volatile device atomic_uint*)input;
    return atomic_fetch_or_explicit(atomic_input, M, memory_order_relaxed);
}

inline uint atomic_fetch_and_add_uint32(device uint* input, int M) {
    auto atomic_input = (volatile device atomic_uint*)input;
    return atomic_fetch_add_explicit(atomic_input, M, memory_order_relaxed);
}

inline uint atomic_fetch_and_dec_uint32(device uint* input) {
    return atomic_fetch_and_add_uint32(input, -1);
}

inline uint atomic_fetch_and_inc_uint32(device uint* input) {
    return atomic_fetch_and_add_uint32(input, 1);
}

/* w0, w1, w2, and w3 are the four cubic B-spline basis functions. */
inline float cubic_w0(float a)
{
  return (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
}

inline float cubic_w1(float a)
{
  return (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
}

inline float cubic_w2(float a)
{
  return (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
}

inline float cubic_w3(float a)
{
  return (1.0f / 6.0f) * (a * a * a);
}

/* g0 and g1 are the two amplitude functions. */
inline float cubic_g0(float a)
{
  return cubic_w0(a) + cubic_w1(a);
}

inline float cubic_g1(float a)
{
  return cubic_w2(a) + cubic_w3(a);
}

/* h0 and h1 are the two offset functions */
inline float cubic_h0(float a)
{
  /* Note +0.5 offset to compensate for CUDA linear filtering convention. */
  return -1.0f + cubic_w1(a) / (cubic_w0(a) + cubic_w1(a)) + 0.5f;
}

inline float cubic_h1(float a)
{
  return 1.0f + cubic_w3(a) / (cubic_w2(a) + cubic_w3(a)) + 0.5f;
}

#define __SPLIT_KERNEL__

#include "kernel/kernel_math.h"
#include "kernel/kernel_types.h"

#include "kernel/split/kernel_split_data.h"

#include "kernel/kernel_globals.h"
#include "kernel/kernel_color.h"

constexpr sampler s(coord::normalized,
                    address::repeat,
                    filter::linear);

/* Fast bicubic texture lookup using 4 bilinear lookups, adapted from CUDA samples. */
template<typename T, typename U>
inline T
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
inline T kernel_tex_image_interp_bicubic_3d(
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



inline float4 kernel_tex_image_interp_3d(thread KernelGlobals *kg,
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
      float4 f;

      if (interpolation == INTERPOLATION_CUBIC) {
        f = kernel_tex_image_interp_bicubic_3d<float4, float>(info, tex, x, y, z);
      }
      else {
          f = tex.sample(s, P);
      }

        return float4(f.x, f.x, f.x, 1.0f);
    }

}

inline float4 kernel_tex_image_interp(thread KernelGlobals *kg, int id, float x, float y) {

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
        float4 f;

        if (info.interpolation == INTERPOLATION_CUBIC) {
            f = kernel_tex_image_interp_bicubic<float4, float>(info, tex, x, y);
        }
        else {
            f = tex.sample(s, P);
        }

        return float4(f.x, f.x, f.x, 1.0f);
    }
}

#include "kernel/split/kernel_split_common.h"
//#include "kernel/split/kernel_data_init.h"
//#include "kernel/split/kernel_path_init.h"
//#include "kernel/split/kernel_scene_intersect.h"
//#include "kernel/split/kernel_lamp_emission.h"
//#include "kernel/split/kernel_do_volume.h"
//#include "kernel/split/kernel_queue_enqueue.h"
//#include "kernel/split/kernel_indirect_background.h"
//#include "kernel/split/kernel_shader_setup.h"
//#include "kernel/split/kernel_shader_sort.h"
//#include "kernel/split/kernel_shader_eval.h"
//#include "kernel/split/kernel_holdout_emission_blurring_pathtermination_ao.h"
//#include "kernel/split/kernel_subsurface_scatter.h"
//#include "kernel/split/kernel_direct_lighting.h"
//#include "kernel/split/kernel_shadow_blocked_ao.h"
//#include "kernel/split/kernel_shadow_blocked_dl.h"
//#include "kernel/split/kernel_enqueue_inactive.h"
//#include "kernel/split/kernel_next_iteration_setup.h"
//#include "kernel/split/kernel_indirect_subsurface.h"
//#include "kernel/split/kernel_buffer_update.h"
//#include "kernel/split/kernel_adaptive_stopping.h"
//#include "kernel/split/kernel_adaptive_filter_x.h"
//#include "kernel/split/kernel_adaptive_filter_y.h"
//#include "kernel/split/kernel_adaptive_adjust_samples.h"


#ifdef __SHADER_RAYTRACE__
#undef __SHADER_RAYTRACE__
#endif

#define __KERNEL_OPTIX__

//#include "bvh/bvh.h"

#undef __KERNEL_OPTIX__

//#undef __VOLUME__

//#include "kernel/kernel_montecarlo.h"
//#include "kernel/kernel_projection.h"
//#include "kernel/kernel_path.h"
//
#undef __BRANCHED_PATH__
//#include "kernel/kernel_bake.h"
//
//#include "kernel/kernel_work_stealing.h"
//
//kernel void kernel_metal_background(device uint4 *input [[buffer(0)]],
//                                    device float4 *output [[buffer(1)]],
//                                    device int *type [[buffer(2)]],
//                                    device int *filterType [[buffer(3)]],
//                                    uint2 grid_size [[grid_size]],
//                                    uint2 thread_position [[thread_position_in_grid]]) {
//    uint x = thread_position.x;
//
//    if (x < grid_size.x) {
//        // do stuff
//        KernelGlobals kg;
//        kernel_bake_evaluate(&kg, input, output, (ShaderEvalType)*type, *filterType, x, 0, 0);
//    }
//}
//
//kernel void kernel_metal_path_trace(device WorkTile* tile [[buffer(0)]],
//                                    uint2 total_work_size_2d [[grid_size]],
//                                    uint3 thread_id [[thread_position_in_grid]]) {
//    uint total_work_size = total_work_size_2d.x;
//    uint work_index = thread_id.x; //ccl_global_id(0);
//    bool thread_is_active = work_index < total_work_size;
//    uint x, y, sample;
//    KernelGlobals kg;
//    if(thread_is_active) {
//        get_work_pixel(tile, work_index, &x, &y, &sample);
//
//        kernel_path_trace(&kg, tile->buffer, sample, x, y, tile->offset, tile->stride);
//    }
//
//    if(kg.data->film.cryptomatte_passes) {
//        threadgroup_barrier(mem_flags::mem_none);
//        if(thread_is_active) {
//            kernel_cryptomatte_post(&kg, tile->buffer, sample, x, y, tile->offset, tile->stride);
//        }
//    }
//}



//#include "util/util_transform.h"
//#include "util/util_atomic.h"
//#include "kernel/kernel_math.h"
//#include "kernel/kernel_types.h"
//#include "util/util_half.h"
//#include "kernel/kernel_globals.h"
//#include "kernel/kernel_color.h"
//
//#include "kernel/kernel_film.h"
