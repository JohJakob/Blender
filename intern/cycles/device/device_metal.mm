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

CCL_NAMESPACE_BEGIN

class MetalDevice : public Device {
 public:
  MetalDevice(DeviceInfo &info, Stats &stats, Profiler &profiler, bool background_)
      : Device(info, stats, profiler, background_)
  {
  }

  virtual ~MetalDevice()
  {
  }

  virtual bool show_samples() const
  {
    return false;
  }

  virtual void const_copy_to(const char *name, void *host, size_t size)
  {
  }

  virtual BVHLayoutMask get_bvh_layout_mask() const
  {
    return BVH_LAYOUT_NONE;
  };

  virtual void task_add(DeviceTask &task){};
  virtual void task_wait(){};
  virtual void task_cancel(){};

 protected:
  virtual void mem_alloc(device_memory &mem){};
  virtual void mem_copy_to(device_memory &mem){};
  virtual void mem_copy_from(device_memory &mem, int y, int w, int h, int elem){};
  virtual void mem_zero(device_memory &mem){};
  virtual void mem_free(device_memory &mem){};
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

CCL_NAMESPACE_END
