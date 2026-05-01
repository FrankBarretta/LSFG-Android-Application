#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lsfg_android {

// Error codes returned to Kotlin via JNI. Keep kOk == 0.
constexpr int kOk = 0;
constexpr int kErrDllUnreadable = -1;
constexpr int kErrMissingResource = -2;
constexpr int kErrTranslationFailed = -3;
constexpr int kErrWriteFailed = -4;

// Parses Lossless.dll at [dllPath], extracts every RCDATA resource that LSFG
// uses, and writes one file per resource. The DXBC FP32 set (IDs 255..302)
// is translated DXBC→SPIR-V and written to <cacheDir>/<resId>.spv.
// The SPIR-V FP16 set (IDs 304..351) is precompiled inside the DLL — those
// blobs are written verbatim to <cacheDir>/fp16/<resId>.spv with no translation.
// The FP16 set is best-effort: if any blob is missing or has the wrong magic
// the FP16 cache is skipped without failing the FP32 extraction.
int extract_dll_to_spirv(const std::string &dllPath, const std::string &cacheDir);

// Reads a cached SPIR-V file back from disk. Returns an empty vector on
// missing/unreadable files — the caller decides whether that's a fatal error.
// Pass useFp16=true to read from the <cacheDir>/fp16/ subdirectory.
std::vector<uint8_t> load_cached_spirv(const std::string &cacheDir, uint32_t resId,
                                       bool useFp16 = false);

// Maps a framegen shader name (e.g. "p_mipmaps", "p_alpha[2]", "generate")
// to the DXBC resource ID (255..302) that lives on disk as <cacheDir>/<id>.spv.
// Returns 0 if the name is unknown.
//
// Mirror of Extract::nameIdxTable in lsfg-vk-android/src/extract/extract.cpp.
uint32_t shader_name_to_resource_id(const std::string &name);

// Same lookup, but returning the SPIR-V FP16 resource ID (304..351). The FP16
// set is a parallel SPIR-V variant precompiled into Lossless.dll with the
// `OpCapability Float16` enabled and mixed FP16/FP32 ops. The mapping is a
// constant +49 offset over shader_name_to_resource_id(). Returns 0 if the
// name is unknown OR if the corresponding FP16 ID is not in the supported
// range (currently 304..351).
uint32_t shader_name_to_resource_id_fp16(const std::string &name);

// Returns true when the FP16 cache directory contains every shader in the
// 304..351 range. Used by the render loop to fall back to the DXBC FP32 path
// transparently when the user toggles FP16 on but the DLL extraction skipped
// the FP16 set (e.g. for an older DLL build that doesn't include them).
bool fp16_shaders_available(const std::string &cacheDir);

} // namespace lsfg_android
