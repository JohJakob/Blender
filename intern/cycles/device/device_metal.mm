/*
 * Copyright 2011-2013 Blender Foundation
 *
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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "device/device_metal.h"

#include "device/device.h"
#include "device/device_intern.h"

#include "util/util_time.h"

CCL_NAMESPACE_BEGIN

class MetalDevice : public Device {
private:

  struct Camera {
      vector_float3 position;
      vector_float3 right;
      vector_float3 up;
      vector_float3 forward;
  };

  struct AreaLight {
      vector_float3 position;
      vector_float3 forward;
      vector_float3 right;
      vector_float3 up;
      vector_float3 color;
  };

  struct Uniforms
  {
      unsigned int width;
      unsigned int height;
      unsigned int frameIndex;
      Camera camera;
      AreaLight light;
  };

  const NSUInteger maxFramesInFlight = 3;
  const size_t alignedUniformsSize = (sizeof(Uniforms) + 255) & ~255;

  const size_t rayStride = 48;
  const size_t intersectionStride = sizeof(MPSIntersectionDistancePrimitiveIndexCoordinates);

  DedicatedTaskPool task_pool;

  class MetalDeviceTask : public DeviceTask {
   public:
    MetalDeviceTask(MetalDevice *device, DeviceTask &task) : DeviceTask(task)
    {
      run = function_bind(&MetalDevice::thread_run, device, this);
    }
  };

  id<MTLDevice> metalDevice;
  id<MTLCommandQueue> commandQueue;
  id<MTLLibrary> library;

  MPSTriangleAccelerationStructure *_accelerationStructure;
  MPSRayIntersector *_intersector;

  id <MTLBuffer> _vertexPositionBuffer;
  id <MTLBuffer> _vertexNormalBuffer;
  id <MTLBuffer> _vertexColorBuffer;
  id <MTLBuffer> _rayBuffer;
  id <MTLBuffer> _shadowRayBuffer;
  id <MTLBuffer> _intersectionBuffer;
  id <MTLBuffer> _uniformBuffer;
  id <MTLBuffer> _triangleMaskBuffer;

  id <MTLComputePipelineState> _rayPipeline;
  id <MTLComputePipelineState> _shadePipeline;
  id <MTLComputePipelineState> _shadowPipeline;
  id <MTLComputePipelineState> _accumulatePipeline;
  id <MTLRenderPipelineState> _copyPipeline;

  id <MTLTexture> _renderTargets[2];
  id <MTLTexture> _accumulationTargets[2];
  id <MTLTexture> _randomTexture;

  dispatch_semaphore_t _sem;
  CGSize _size;
  NSUInteger _uniformBufferOffset;
  NSUInteger _uniformBufferIndex;

  unsigned int _frameIndex;

  void createPipelines() {
    NSError *error = NULL;

    // Create compute pipelines will will execute code on the GPU
    MTLComputePipelineDescriptor *computeDescriptor = [[MTLComputePipelineDescriptor alloc] init];

    // Set to YES to allow compiler to make certain optimizations
    computeDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;

    // Generates rays according to view/projection matrices
    computeDescriptor.computeFunction = [library newFunctionWithName:@"rayKernel"];

    _rayPipeline = [metalDevice newComputePipelineStateWithDescriptor:computeDescriptor
                                                          options:0
                                                       reflection:nil
                                                            error:&error];

    if (!_rayPipeline)
        NSLog(@"Failed to create pipeline state: %@", error);

    // Consumes ray/scene intersection test results to perform shading
    computeDescriptor.computeFunction = [library newFunctionWithName:@"shadeKernel"];

    _shadePipeline = [metalDevice newComputePipelineStateWithDescriptor:computeDescriptor
                                                          options:0
                                                       reflection:nil
                                                            error:&error];

    if (!_shadePipeline)
        NSLog(@"Failed to create pipeline state: %@", error);

    // Consumes shadow ray intersection tests to update the output image
    computeDescriptor.computeFunction = [library newFunctionWithName:@"shadowKernel"];

    _shadowPipeline = [metalDevice newComputePipelineStateWithDescriptor:computeDescriptor
                                                             options:0
                                                          reflection:nil
                                                               error:&error];

    if (!_shadowPipeline)
        NSLog(@"Failed to create pipeline state: %@", error);

    // Averages the current frame's output image with all previous frames
    computeDescriptor.computeFunction = [library newFunctionWithName:@"accumulateKernel"];

    _accumulatePipeline = [metalDevice newComputePipelineStateWithDescriptor:computeDescriptor
                                                                 options:0
                                                              reflection:nil
                                                                   error:&error];

    if (!_accumulatePipeline)
        NSLog(@"Failed to create pipeline state: %@", error);

  }

  void createBuffers() {
    // Uniform buffer contains a few small values which change from frame to frame. We will have up to 3
    // frames in flight at once, so allocate a range of the buffer for each frame. The GPU will read from
    // one chunk while the CPU writes to the next chunk. Each chunk must be aligned to 256 bytes on macOS
    // and 16 bytes on iOS.
    NSUInteger uniformBufferSize = alignedUniformsSize * maxFramesInFlight;

    // Vertex data should be stored in private or managed buffers on discrete GPU systems (AMD, NVIDIA).
    // Private buffers are stored entirely in GPU memory and cannot be accessed by the CPU. Managed
    // buffers maintain a copy in CPU memory and a copy in GPU memory.
    MTLResourceOptions options = 0;

    options = MTLResourceStorageModeManaged;

    _uniformBuffer = [metalDevice newBufferWithLength:uniformBufferSize options:options];

    // Allocate buffers for vertex positions, colors, and normals. Note that each vertex position is a
    // float3, which is a 16 byte aligned type.
//    _vertexPositionBuffer = [metalDevice newBufferWithLength:vertices.size() * sizeof(float3) options:options];
//    _vertexColorBuffer = [metalDevice newBufferWithLength:colors.size() * sizeof(float3) options:options];
//    _vertexNormalBuffer = [metalDevice newBufferWithLength:normals.size() * sizeof(float3) options:options];
//    _triangleMaskBuffer = [metalDevice newBufferWithLength:masks.size() * sizeof(uint32_t) options:options];

    // Copy vertex data into buffers
//    memcpy(_vertexPositionBuffer.contents, &vertices[0], _vertexPositionBuffer.length);
//    memcpy(_vertexColorBuffer.contents, &colors[0], _vertexColorBuffer.length);
//    memcpy(_vertexNormalBuffer.contents, &normals[0], _vertexNormalBuffer.length);
//    memcpy(_triangleMaskBuffer.contents, &masks[0], _triangleMaskBuffer.length);

    // When using managed buffers, we need to indicate that we modified the buffer so that the GPU
    // copy can be updated
    [_vertexPositionBuffer didModifyRange:NSMakeRange(0, _vertexPositionBuffer.length)];
    [_vertexColorBuffer didModifyRange:NSMakeRange(0, _vertexColorBuffer.length)];
    [_vertexNormalBuffer didModifyRange:NSMakeRange(0, _vertexNormalBuffer.length)];
    [_triangleMaskBuffer didModifyRange:NSMakeRange(0, _triangleMaskBuffer.length)];

  }

  void createIntersector() {
    // Create a raytracer for our Metal device
    _intersector = [[MPSRayIntersector alloc] initWithDevice:metalDevice];
    

    _intersector.rayDataType = MPSRayDataTypeOriginMaskDirectionMaxDistance;
    _intersector.rayStride = rayStride;
    _intersector.rayMaskOptions = MPSRayMaskOptionPrimitive;

    // Create an acceleration structure from our vertex position data
    _accelerationStructure = [[MPSTriangleAccelerationStructure alloc] initWithDevice:metalDevice];

    _accelerationStructure.vertexBuffer = _vertexPositionBuffer;
    _accelerationStructure.maskBuffer = _triangleMaskBuffer;
//    _accelerationStructure.triangleCount = vertices.size() / 3;

    [_accelerationStructure rebuild];

  }

  void updateUniforms() {
        // Update this frame's uniforms
        _uniformBufferOffset = alignedUniformsSize * _uniformBufferIndex;

        Uniforms *uniforms = (Uniforms *)((char *)_uniformBuffer.contents + _uniformBufferOffset);

//        uniforms->camera.position = vector3(0.0f, 1.0f, 3.38f);
//
//        uniforms->camera.forward = vector3(0.0f, 0.0f, -1.0f);
//        uniforms->camera.right = vector3(1.0f, 0.0f, 0.0f);
//        uniforms->camera.up = vector3(0.0f, 1.0f, 0.0f);
//
//        uniforms->light.position = vector3(0.0f, 1.98f, 0.0f);
//        uniforms->light.forward = vector3(0.0f, -1.0f, 0.0f);
//        uniforms->light.right = vector3(0.25f, 0.0f, 0.0f);
//        uniforms->light.up = vector3(0.0f, 0.0f, 0.25f);
//        uniforms->light.color = vector3(4.0f, 4.0f, 4.0f);

        float fieldOfView = 45.0f * (M_PI / 180.0f);
        float aspectRatio = (float)_size.width / (float)_size.height;
        float imagePlaneHeight = tanf(fieldOfView / 2.0f);
        float imagePlaneWidth = aspectRatio * imagePlaneHeight;

        uniforms->camera.right *= imagePlaneWidth;
        uniforms->camera.up *= imagePlaneHeight;

        uniforms->width = (unsigned int)_size.width;
        uniforms->height = (unsigned int)_size.height;

        uniforms->frameIndex = _frameIndex++;

        [_uniformBuffer didModifyRange:NSMakeRange(_uniformBufferOffset, alignedUniformsSize)];

        // Advance to the next slot in the uniform buffer
        _uniformBufferIndex = (_uniformBufferIndex + 1) % maxFramesInFlight;

  }

  void draw() {
    /*
     // We are using the uniform buffer to stream uniform data to the GPU, so we need to wait until the oldest
     // GPU frame has completed before we can reuse that space in the buffer.
     dispatch_semaphore_wait(_sem, DISPATCH_TIME_FOREVER);

     // Create a command buffer which will contain our GPU commands
     id <MTLCommandBuffer> commandBuffer = [_queue commandBuffer];

     // When the frame has finished, signal that we can reuse the uniform buffer space from this frame.
     // Note that the contents of completion handlers should be as fast as possible as the GPU driver may
     // have other work scheduled on the underlying dispatch queue.
     [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
         dispatch_semaphore_signal(self->_sem);
     }];

     [self updateUniforms];

     NSUInteger width = (NSUInteger)_size.width;
     NSUInteger height = (NSUInteger)_size.height;

     // We will launch a rectangular grid of threads on the GPU to generate the rays. Threads are launched in
     // groups called "threadgroups". We need to align the number of threads to be a multiple of the threadgroup
     // size. We indicated when compiling the pipeline that the threadgroup size would be a multiple of the thread
     // execution width (SIMD group size) which is typically 32 or 64 so 8x8 is a safe threadgroup size which
     // should be small to be supported on most devices. A more advanced application would choose the threadgroup
     // size dynamically.
     MTLSize threadsPerThreadgroup = MTLSizeMake(8, 8, 1);
     MTLSize threadgroups = MTLSizeMake((width  + threadsPerThreadgroup.width  - 1) / threadsPerThreadgroup.width,
                                        (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                        1);

     // First, we will generate rays on the GPU. We create a compute command encoder which will be used to add
     // commands to the command buffer.
     id <MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

     // Bind buffers needed by the compute pipeline
     [computeEncoder setBuffer:_uniformBuffer   offset:_uniformBufferOffset atIndex:0];
     [computeEncoder setBuffer:_rayBuffer       offset:0                    atIndex:1];

     [computeEncoder setTexture:_randomTexture    atIndex:0];
     [computeEncoder setTexture:_renderTargets[0] atIndex:1];

     // Bind the ray generation compute pipeline
     [computeEncoder setComputePipelineState:_rayPipeline];

     // Launch threads
     [computeEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];

     // End the encoder
     [computeEncoder endEncoding];

     // We will iterate over the next few kernels several times to allow light to bounce around the scene
     for (int bounce = 0; bounce < 12; bounce++) {
         _intersector.intersectionDataType = MPSIntersectionDataTypeDistancePrimitiveIndexCoordinates;

         // We can then pass the rays to the MPSRayIntersector to compute the intersections with our acceleration structure
         [_intersector encodeIntersectionToCommandBuffer:commandBuffer               // Command buffer to encode into
                                        intersectionType:MPSIntersectionTypeNearest  // Intersection test type
                                               rayBuffer:_rayBuffer                  // Ray buffer
                                         rayBufferOffset:0                           // Offset into ray buffer
                                      intersectionBuffer:_intersectionBuffer         // Intersection buffer (destination)
                                intersectionBufferOffset:0                           // Offset into intersection buffer
                                                rayCount:width * height              // Number of rays
                                   accelerationStructure:_accelerationStructure];    // Acceleration structure
         // We launch another pipeline to consume the intersection results and shade the scene
         computeEncoder = [commandBuffer computeCommandEncoder];

         [computeEncoder setBuffer:_uniformBuffer      offset:_uniformBufferOffset atIndex:0];
         [computeEncoder setBuffer:_rayBuffer          offset:0                    atIndex:1];
         [computeEncoder setBuffer:_shadowRayBuffer    offset:0                    atIndex:2];
         [computeEncoder setBuffer:_intersectionBuffer offset:0                    atIndex:3];
         [computeEncoder setBuffer:_vertexColorBuffer  offset:0                    atIndex:4];
         [computeEncoder setBuffer:_vertexNormalBuffer offset:0                    atIndex:5];
         [computeEncoder setBuffer:_triangleMaskBuffer offset:0                    atIndex:6];
         [computeEncoder setBytes:&bounce              length:sizeof(bounce)       atIndex:7];

         [computeEncoder setTexture:_randomTexture    atIndex:0];
         [computeEncoder setTexture:_renderTargets[0] atIndex:1];

         [computeEncoder setComputePipelineState:_shadePipeline];

         [computeEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];

         [computeEncoder endEncoding];

         // We intersect rays with the scene, except this time we are intersecting shadow rays. We only need
         // to know whether the shadows rays hit anything on the way to the light source, not which triangle
         // was intersected. Therefore, we can use the "any" intersection type to end the intersection search
         // as soon as any intersection is found. This is typically much faster than finding the nearest
         // intersection. We can also use MPSIntersectionDataTypeDistance, because we don't need the triangle
         // index and barycentric coordinates.
         _intersector.intersectionDataType = MPSIntersectionDataTypeDistance;

         [_intersector encodeIntersectionToCommandBuffer:commandBuffer
                                        intersectionType:MPSIntersectionTypeAny
                                               rayBuffer:_shadowRayBuffer
                                         rayBufferOffset:0
                                      intersectionBuffer:_intersectionBuffer
                                intersectionBufferOffset:0
                                                rayCount:width * height
                                   accelerationStructure:_accelerationStructure];

         // Finally, we launch a kernel which writes the color computed by the shading kernel into the
         // output image, but only if the corresponding shadow ray does not intersect anything on the way to
         // the light. If the shadow ray intersects a triangle before reaching the light source, the original
         // intersection point was in shadow.
         computeEncoder = [commandBuffer computeCommandEncoder];

         [computeEncoder setBuffer:_uniformBuffer      offset:_uniformBufferOffset atIndex:0];
         [computeEncoder setBuffer:_shadowRayBuffer    offset:0                    atIndex:1];
         [computeEncoder setBuffer:_intersectionBuffer offset:0                    atIndex:2];

         [computeEncoder setTexture:_renderTargets[0] atIndex:0];
         [computeEncoder setTexture:_renderTargets[1] atIndex:1];

         [computeEncoder setComputePipelineState:_shadowPipeline];

         [computeEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];

         [computeEncoder endEncoding];

         std::swap(_renderTargets[0], _renderTargets[1]);
     }

     // The final kernel averages the current frame's image with all previous frames to reduce noise due
     // random sampling of the scene.
     computeEncoder = [commandBuffer computeCommandEncoder];

     [computeEncoder setBuffer:_uniformBuffer      offset:_uniformBufferOffset atIndex:0];

     [computeEncoder setTexture:_renderTargets[0]       atIndex:0];
     [computeEncoder setTexture:_accumulationTargets[0] atIndex:1];
     [computeEncoder setTexture:_accumulationTargets[1] atIndex:2];

     [computeEncoder setComputePipelineState:_accumulatePipeline];

     [computeEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];

     [computeEncoder endEncoding];

     std::swap(_accumulationTargets[0], _accumulationTargets[1]);

     // Copy the resulting image into our view using the graphics pipeline since we can't write directly to
     // it with a compute kernel. We need to delay getting the current render pass descriptor as long as
     // possible to avoid stalling until the GPU/compositor release a drawable. The render pass descriptor
     // may be nil if the window has moved off screen.
     MTLRenderPassDescriptor* renderPassDescriptor = view.currentRenderPassDescriptor;

     if (renderPassDescriptor != nil) {
         // Create a render encoder
         id <MTLRenderCommandEncoder> renderEncoder =
         [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];

         [renderEncoder setRenderPipelineState:_copyPipeline];

         [renderEncoder setFragmentTexture:_accumulationTargets[0] atIndex:0];

         // Draw a quad which fills the screen
         [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];

         [renderEncoder endEncoding];

         // Present the drawable to the screen
         [commandBuffer presentDrawable:view.currentDrawable];
     }

     // Finally, commit the command buffer so that the GPU can start executing
     [commandBuffer commit];

     */
  }

 public:

  MetalDevice(DeviceInfo &info, Stats &stats, Profiler &profiler, bool background_)
      : Device(info, stats, profiler, background_)
  {
    // TODO: grab the specific device that was asked for
    metalDevice = MTLCreateSystemDefaultDevice();
    commandQueue = [metalDevice newCommandQueue];
    library = [metalDevice newDefaultLibrary];

    NSLog(@"Metal device: %@", metalDevice.name);

    _sem = dispatch_semaphore_create(maxFramesInFlight);

    createPipelines();
    createBuffers();
    createIntersector();
  }

  virtual ~MetalDevice()
  {
    [library release];
    [commandQueue release];
    [metalDevice release];
  }

  virtual bool show_samples() const
  {
    return false;
  }

  virtual void const_copy_to(const char *name, void *host, size_t size)
  {
    std::cout << "const_copy_to does jack squat on Metal" << std::endl;
    std::cout << "name: " << name << " size: " << size << std::endl;
  }

  virtual BVHLayoutMask get_bvh_layout_mask() const
  {
    return BVH_LAYOUT_NONE;
  };

  void task_add(DeviceTask &task)
  {
    task_pool.push(new MetalDeviceTask(this, task));
  }

  void task_wait()
  {
    task_pool.wait();
  }

  void task_cancel()
  {
    task_pool.cancel();
  }

  void thread_run(DeviceTask *task)
  {
    // todo flush_texture_buffers

    if (task->type == DeviceTask::RENDER || task->type == DeviceTask::DENOISE_BUFFER) {
      RenderTile tile;
      DenoisingTask denoising(this, *task);

      /* Allocate buffer for kernel globals */

      /* Keep rendering tiles until done. */
      while (task->acquire_tile(this, tile, task->tile_types)) {
        if (tile.task == RenderTile::PATH_TRACE) {
          assert(tile.task == RenderTile::PATH_TRACE);
          scoped_timer timer(&tile.buffers->render_time);

  //        split_kernel->path_trace(task, tile, kgbuffer, *const_mem_map["__data"]);

  //        clFinish(cqCommandQueue); ???
        }
        else if (tile.task == RenderTile::DENOISE) {
          tile.sample = tile.start_sample + tile.num_samples;
  //        denoise(tile, denoising);
          task->update_progress(&tile, tile.w * tile.h);
        }

        task->release_tile(tile);
      }

  //    kgbuffer.free();
    }
    else if (task->type == DeviceTask::SHADER) {
  //    shader(*task);
    }
    else if (task->type == DeviceTask::FILM_CONVERT) {
  //    film_convert(*task, task->buffer, task->rgba_byte, task->rgba_half);
    }
    else if (task->type == DeviceTask::DENOISE_BUFFER) {
      RenderTile tile;
      tile.x = task->x;
      tile.y = task->y;
      tile.w = task->w;
      tile.h = task->h;
      tile.buffer = task->buffer;
      tile.sample = task->sample + task->num_samples;
      tile.num_samples = task->num_samples;
      tile.start_sample = task->sample;
      tile.offset = task->offset;
      tile.stride = task->stride;
      tile.buffers = task->buffers;

      DenoisingTask denoising(this, *task);
  //    denoise(tile, denoising);
      task->update_progress(&tile, tile.w * tile.h);
    }
  }

 protected:
  virtual void mem_alloc(device_memory &mem)
  {

    NSUInteger len = mem.memory_size();

    MTLResourceOptions options = MTLResourceStorageModeManaged;

    id<MTLBuffer> buffer = [metalDevice newBufferWithLength:len options:options];


    if (@available(macOS 10.13, *)) {
      mem.device_size = buffer.allocatedSize;
    } else {
      mem.device_size = buffer.length;
    }
    mem.device_pointer = (device_ptr)buffer;
  };
  virtual void mem_copy_to(device_memory &mem)
  {
    id<MTLBuffer> b = (id<MTLBuffer>)mem.device_pointer;
    NSRange entireBuffer = NSMakeRange(0, mem.memory_size());
    [b didModifyRange:entireBuffer];
  };
  virtual void mem_copy_from(device_memory &mem, int y, int w, int h, int elem)
  {
    id<MTLCommandBuffer> cb = [commandQueue commandBuffer];

    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

    id<MTLBuffer> b = (id<MTLBuffer>)mem.device_pointer;

    [blit synchronizeResource:b];

    [cb commit];
  };
  virtual void mem_zero(device_memory &mem)
  {
    id<MTLCommandBuffer> b = [commandQueue commandBuffer];

    id<MTLBlitCommandEncoder> blit = [b blitCommandEncoder];

    NSRange everything = NSMakeRange(0, mem.device_size);
    [blit fillBuffer:(id<MTLBuffer>)mem.device_pointer
               range:everything
               value:0];

    [blit endEncoding];

    [b commit];

  };
  virtual void mem_free(device_memory &mem)
  {
    mem.device_pointer = NULL;
  };
};

Device *device_metal_create(DeviceInfo &info, Stats &stats, Profiler &profiler, bool background)
{
  MetalDevice *md = new MetalDevice(info, stats, profiler, background);

  return md;
}

void device_metal_info(vector<DeviceInfo> &devices)
{
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();

  DeviceInfo mdi;
  mdi.type = DEVICE_METAL;
  mdi.id = "METAL_DEVICE_DEFAULT";
  mdi.description = [device.name cStringUsingEncoding:NSUTF8StringEncoding];
  devices.push_back(mdi);
}

bool device_metal_init()
{
  return true;
}

string device_metal_capabilities()
{
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();

  return [device.description cStringUsingEncoding:NSUTF8StringEncoding];
}

CCL_NAMESPACE_END
