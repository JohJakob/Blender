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

 public:

  MetalDevice(DeviceInfo &info, Stats &stats, Profiler &profiler, bool background_)
      : Device(info, stats, profiler, background_)
  {
    // TODO: grab the specific device that was asked for
    metalDevice = MTLCreateSystemDefaultDevice();
    commandQueue = [metalDevice newCommandQueue];
    NSURL *libURL = [NSBundle.mainBundle URLForResource:@"Shaders" withExtension:@"metallib"];
    NSError *error;
    id<MTLLibrary> lib = [metalDevice newLibraryWithURL:libURL error:&error];
    library = lib;

    setupPipelines();

      NSLog(@"done with metal setup");
  }

  virtual ~MetalDevice()
  {
    [library release];
    [commandQueue release];
    [metalDevice release];
  }

    bool show_samples() const override
  {
    return false;
  }

  void const_copy_to(const char *name, void *host, size_t size) override
  {
    std::cout << "const_copy_to does jack squat on Metal" << std::endl;
    std::cout << "name: " << name << " size: " << size << std::endl;
  }

  BVHLayoutMask get_bvh_layout_mask() const override
  {
    return BVH_LAYOUT_NONE;
  };

  void task_add(DeviceTask &task) override
  {
    task_pool.push(new MetalDeviceTask(this, task));
  }

  void task_wait() override
  {
    task_pool.wait();
  }

  void task_cancel() override
  {
    task_pool.cancel();
  }

    void setupPipelines() {
        commandQueue = [metalDevice newCommandQueue];

        setupBackgroundShaderPipeline();
        setupPathTraceShaderPipeline();
    }

    id<MTLComputePipelineState> pathTraceShaderPipeline;
    void setupPathTraceShaderPipeline() {
        id<MTLFunction> pathTraceKernel = [library newFunctionWithName:@"kernel_split_path_trace"];

        MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
        desc.computeFunction = pathTraceKernel;

        NSError *error;

        pathTraceShaderPipeline = [metalDevice newComputePipelineStateWithDescriptor:desc options:0 reflection:nil error:&error];
    }

    id<MTLComputePipelineState> backgroundPipelineState;
    void setupBackgroundShaderPipeline() {
        id<MTLFunction> backgroundKernel = [library newFunctionWithName:@"kernel_metal_background"];
        MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
        desc.computeFunction = backgroundKernel;

        NSError *error;

        backgroundPipelineState = [metalDevice newComputePipelineStateWithDescriptor:desc options:0 reflection:nil error:&error];
    }

    void shader(DeviceTask &task) {

        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];

        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        if (task.shader_eval_type == SHADER_EVAL_BACKGROUND) {
            // launch background shading kernel

            [enc setComputePipelineState:backgroundPipelineState];

            id<MTLBuffer> inputBuffer = (id<MTLBuffer>)task.shader_input;
            id<MTLBuffer> outputBuffer = (id<MTLBuffer>)task.shader_output;

            int type = task.shader_eval_type;
            int filterType = task.shader_filter;

            [enc setBuffer:inputBuffer offset:0 atIndex:0];
            [enc setBuffer:outputBuffer offset:0 atIndex:1];
            [enc setBytes:&type length:sizeof(type) atIndex:2];
            [enc setBytes:&filterType length:sizeof(filterType) atIndex:3];

            NSUInteger threadgroupMax = backgroundPipelineState.maxTotalThreadsPerThreadgroup;

            NSUInteger groups = (task.shader_w + threadgroupMax - 1) / threadgroupMax;

            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadgroupMax, 1, 1)];

            [enc endEncoding];

            [cmdBuf enqueue];

            [cmdBuf waitUntilCompleted];

            task.update_progress(NULL);

        }

    }

    void path_trace(DeviceTask &task,
                    RenderTile &rtile,
                    device_vector<WorkTile> &work_tiles) {
        scoped_timer timer(&rtile.buffers->render_time);

        if (have_error())
          return;

        /* Allocate work tile. */
        work_tiles.alloc(1);

        WorkTile *wtile = work_tiles.data();
        wtile->x = rtile.x;
        wtile->y = rtile.y;
        wtile->w = rtile.w;
        wtile->h = rtile.h;
        wtile->offset = rtile.offset;
        wtile->stride = rtile.stride;
        wtile->buffer = (float *)rtile.buffer;


        NSUInteger desiredWidth = [pathTraceShaderPipeline threadExecutionWidth];

        /* Render all samples. */
        int start_sample = rtile.start_sample;
        int end_sample = rtile.start_sample + rtile.num_samples;

        for (int sample = start_sample; sample < end_sample; sample += 1) {

            id<MTLCommandBuffer> buff = [commandQueue commandBuffer];

            id<MTLComputeCommandEncoder> enc = [buff computeCommandEncoder];

            [enc setComputePipelineState:pathTraceShaderPipeline];

          /* Setup and copy work tile to device. */
          wtile->start_sample = sample;
          wtile->num_samples = min(1, end_sample - sample);
          work_tiles.copy_to_device();

            id<MTLBuffer> d_work_tiles = (id<MTLBuffer>)work_tiles.device_pointer;
          uint total_work_size = wtile->w * wtile->h * wtile->num_samples;
          uint num_blocks = divide_up(total_work_size, desiredWidth);

            [enc setBuffer:d_work_tiles offset:0 atIndex:0];

            [enc dispatchThreadgroups:{num_blocks, 1, 1} threadsPerThreadgroup:{desiredWidth, 1, 1}];

            [enc endEncoding];

            [buff commit];

            [buff waitUntilCompleted];

          /* Update progress. */
          rtile.sample = sample + wtile->num_samples;
          task.update_progress(&rtile, rtile.w * rtile.h * wtile->num_samples);

          if (task.get_cancel()) {
            if (task.need_finish_queue == false)
              break;
          }
        }
    }

  void thread_run(DeviceTask *task)
  {
      if (task->type == DeviceTask::RENDER) {
          printf("RENDER");
//        DeviceRequestedFeatures requested_features;
//        if (use_split_kernel()) {
//          if (split_kernel == NULL) {
//            split_kernel = new CUDASplitKernel(this);
//            split_kernel->load_kernels(requested_features);
//          }
//        }
//
        device_vector<WorkTile> work_tiles(this, "work_tiles", MEM_READ_ONLY);
//
//        /* keep rendering tiles until done */
        RenderTile tile;
//        DenoisingTask denoising(this, *task);
//
        while (task->acquire_tile(this, tile, task->tile_types)) {
          if (tile.task == RenderTile::PATH_TRACE) {
//            if (use_split_kernel()) {
//              device_only_memory<uchar> void_buffer(this, "void_buffer");
//              split_kernel->path_trace(task, tile, void_buffer, void_buffer);
//            }
//            else {
              path_trace(*task, tile, work_tiles);
//            }
          }
          else if (tile.task == RenderTile::DENOISE) {
            tile.sample = tile.start_sample + tile.num_samples;
//
//            denoise(tile, denoising);
//
            task->update_progress(&tile, tile.w * tile.h);
          }

          task->release_tile(tile);

          if (task->get_cancel()) {
            if (task->need_finish_queue == false)
              break;
          }
        }

        work_tiles.free();
      }
      else if (task->type == DeviceTask::SHADER) {
          printf("SHADER");
          shader(*task);
      }
      else if (task->type == DeviceTask::DENOISE_BUFFER) {
          printf("DENOISE");
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

//        DenoisingTask denoising(this, *task);
//        denoise(tile, denoising);
//        task->update_progress(&tile, tile.w * tile.h);
      }
  }

 protected:
  virtual void mem_alloc(device_memory &mem) override
  {

    NSUInteger len = mem.memory_size();

    MTLResourceOptions options = MTLResourceStorageModeManaged;

    id<MTLBuffer> buffer = [metalDevice newBufferWithLength:len options:options];

    mem.device_size = buffer.length;
    mem.device_pointer = (device_ptr)buffer;
  };
  virtual void mem_copy_to(device_memory &mem) override
  {
    id<MTLBuffer> b = (id<MTLBuffer>)mem.device_pointer;
    NSRange entireBuffer = NSMakeRange(0, mem.memory_size());
    [b didModifyRange:entireBuffer];
  };
  virtual void mem_copy_from(device_memory &mem, int y, int w, int h, int elem) override
  {
    id<MTLCommandBuffer> cb = [commandQueue commandBuffer];

    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

    id<MTLBuffer> b = (id<MTLBuffer>)mem.device_pointer;

    [blit synchronizeResource:b];

    [blit endEncoding];

    [cb commit];

    [cb waitUntilCompleted];
  };
  virtual void mem_zero(device_memory &mem) override
  {
    id<MTLCommandBuffer> b = [commandQueue commandBuffer];

    id<MTLBlitCommandEncoder> blit = [b blitCommandEncoder];

      mem_alloc(mem);
    NSRange everything = NSMakeRange(0, mem.device_size);
    [blit fillBuffer:(id<MTLBuffer>)mem.device_pointer
               range:everything
               value:0];

    [blit endEncoding];

    [b commit];

    [b waitUntilCompleted];

  };
  virtual void mem_free(device_memory &mem) override
  {
    mem.device_pointer = NULL;
  };
};

Device *device_metal_create(DeviceInfo &info, Stats &stats, Profiler &profiler, bool background)
{
  return new MetalDevice(info, stats, profiler, background);
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
