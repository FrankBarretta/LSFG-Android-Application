#pragma once

#include <string>

namespace lsfg_android {

constexpr int kProbeNoVulkan = -10;
constexpr int kProbeMissingSpirv = -11;
constexpr int kProbeDriverRejected = -12;

// Creates a headless Vulkan device and runs vkCreateShaderModule over every
// cached SPIR-V blob. Returns kOk iff all shaders are accepted.
int probe_shaders_on_device(const std::string &cacheDir);

// Reports whether the first physical Vulkan device on this system advertises
// VK_KHR_shader_float16_int8 with shaderFloat16=VK_TRUE. Used by the UI to
// grey out the "FP16 frame-gen shaders" toggle on hardware that can't run
// the OpCapability Float16 SPIR-V variants. Headless: creates a temporary
// VkInstance/VkDevice, checks the feature, and tears down. Returns false on
// any error (no Vulkan loader, no device, etc.) — the UI must treat false
// as "FP16 not available" rather than a failure to probe.
bool device_supports_float16();

} // namespace lsfg_android
