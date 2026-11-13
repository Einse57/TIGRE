/*--------------------------------------------------------------------------
 This file is part of the TIGRE Toolbox

 Copyright (c) 2015, University of Bath and
                     CERN-European Organization for Nuclear Research
                     All rights reserved.

 License:            Open Source under BSD.
                     See the full license at
                     https://github.com/CERN/TIGRE/blob/master/LICENSE

 Contact:            tigre.toolbox@gmail.com
 Codes:              https://github.com/CERN/TIGRE/
 --------------------------------------------------------------------------
 */

#include "tigre_sycl.hpp"
#include <memory>
#include <mutex>
#include <algorithm>

static std::vector<sycl::device> enumerateDevices(){
    // The default SYCL device only (ONEAPI_DEVICE_SELECTOR applies).
    return std::vector<sycl::device>(1, sycl::device(sycl::default_selector_v));
}

const std::vector<sycl::device>& tigreSyclDevices(){
    static const std::vector<sycl::device> devices = enumerateDevices();
    return devices;
}

sycl::queue& tigreSyclQueue(int iDeviceId){
    static std::vector<std::unique_ptr<sycl::queue>> queues;
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    const auto& devices = tigreSyclDevices();
    if (queues.size()!=devices.size())
        queues.resize(devices.size());
    if (!queues[iDeviceId])
        queues[iDeviceId].reset(new sycl::queue(devices[iDeviceId], sycl::property::queue::in_order()));
    return *queues[iDeviceId];
}

size_t tigreSyclMaxAlloc(int iDeviceId){
    const sycl::device& device = tigreSyclDevices()[iDeviceId];
    size_t global   = device.get_info<sycl::info::device::global_mem_size>();
    size_t maxAlloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
    return std::min((size_t)((double)global*0.95), maxAlloc);
}
