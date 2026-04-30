// On-device port of Extract::extractShaders + Extract::translateShader.
//
// On Linux the DLL path is discovered from Steam install locations. On Android
// the user picks the file via SAF and Kotlin copies it to a local path under
// the app's filesDir before calling into native. This file only implements
// the PE parsing + DXBC→SPIR-V translation and caches one .spv per resource
// ID into a caller-chosen cache directory.

#include "android_shader_loader.hpp"

#include <pe-parse/parse.h>

#include <dxbc_modinfo.h>
#include <dxbc_module.h>
#include <dxbc_reader.h>
#include <thirdparty/spirv.hpp>

#include <android/log.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "crash_reporter.hpp"

#define LOG_TAG "lsfg-extract"
#define LOGE(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_ERROR, __VA_ARGS__)
#define LOGI(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_INFO,  __VA_ARGS__)

namespace {

// Resource IDs used by both LSFG 3.1 and the 3.1P "performance" variant.
// Kept in sync with lsfg-vk/src/extract/extract.cpp::nameIdxTable.
constexpr uint32_t kResourceIds[] = {
    255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,
    267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279,
    280, 281, 282, 283, 284, 285, 286, 287, 288, 289,
    290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302,
};

struct ExtractionCtx {
    std::unordered_map<uint32_t, std::vector<uint8_t>> *out;
};

int on_resource(void *userData, const peparse::resource &res) {
    auto *ctx = static_cast<ExtractionCtx *>(userData);
    if (res.type != peparse::RT_RCDATA || res.buf == nullptr || res.buf->bufLen <= 0) {
        return 0;
    }
    std::vector<uint8_t> data(res.buf->bufLen);
    std::copy_n(res.buf->buf, res.buf->bufLen, data.data());
    (*ctx->out)[res.name] = std::move(data);
    return 0;
}

struct BindingOffsets {
    uint32_t bindingIndex{};
    uint32_t bindingOffset{};
    uint32_t setIndex{};
    uint32_t setOffset{};
};

// SPIR-V header layout: [magic][version][generator][bound][schema]
// Vulkan 1.1 supports up to SPIR-V 1.3 (0x00010300). The dxvk compiler
// targets SPIR-V 1.6 which Mali rejects at JIT compilation time. Patch the
// version word down so the driver accepts the bytecode.
void cap_spirv_version(std::vector<uint8_t> &spirv) {
    if (spirv.size() < 20) return;
    uint32_t ver;
    std::memcpy(&ver, spirv.data() + 4, 4);
    if (ver > 0x00010300u) {
        ver = 0x00010300u;
        std::memcpy(spirv.data() + 4, &ver, 4);
    }
}

std::vector<uint8_t> translate_dxbc_to_spirv(const std::vector<uint8_t> &bytecode) {
    dxvk::DxbcReader reader(reinterpret_cast<const char *>(bytecode.data()), bytecode.size());
    dxvk::DxbcModule module(reader);
    const dxvk::DxbcModuleInfo info{};
    auto code = module.compile(info, "CS");

    std::vector<BindingOffsets> bindingOffsets;
    std::vector<uint32_t> varIds;
    for (auto ins : code) {
        if (ins.opCode() == spv::OpDecorate) {
            if (ins.arg(2) == spv::DecorationBinding) {
                const uint32_t varId = ins.arg(1);
                bindingOffsets.resize(std::max(bindingOffsets.size(), size_t(varId + 1)));
                bindingOffsets[varId].bindingIndex = ins.arg(3);
                bindingOffsets[varId].bindingOffset = ins.offset() + 3;
                varIds.push_back(varId);
            }
            if (ins.arg(2) == spv::DecorationDescriptorSet) {
                const uint32_t varId = ins.arg(1);
                bindingOffsets.resize(std::max(bindingOffsets.size(), size_t(varId + 1)));
                bindingOffsets[varId].setIndex = ins.arg(3);
                bindingOffsets[varId].setOffset = ins.offset() + 3;
            }
        }
        if (ins.opCode() == spv::OpFunction) {
            break;
        }
    }

    std::vector<BindingOffsets> validBindings;
    for (const auto varId : varIds) {
        const auto info = bindingOffsets[varId];
        if (info.bindingOffset) {
            validBindings.push_back(info);
        }
    }

    for (size_t i = 0; i < validBindings.size(); ++i) {
        code.data()[validBindings[i].bindingOffset] = static_cast<uint8_t>(i);
    }

    std::vector<uint8_t> spirv(code.size());
    std::copy_n(reinterpret_cast<const uint8_t *>(code.data()), code.size(), spirv.data());
    cap_spirv_version(spirv);
    return spirv;
}

bool write_file(const std::string &path, const std::vector<uint8_t> &data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    f.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    return f.good();
}

} // namespace

namespace lsfg_android {

int extract_dll_to_spirv(const std::string &dllPath, const std::string &cacheDir) {
    peparse::parsed_pe *dll = peparse::ParsePEFromFile(dllPath.c_str());
    if (!dll) {
        LOGE("ParsePEFromFile failed for %s", dllPath.c_str());
        return kErrDllUnreadable;
    }

    std::unordered_map<uint32_t, std::vector<uint8_t>> dxbcByResId;
    ExtractionCtx ctx{&dxbcByResId};
    peparse::IterRsrc(dll, on_resource, &ctx);
    peparse::DestructParsedPE(dll);

    // Ensure every resource ID we care about was present in the DLL.
    for (uint32_t id : kResourceIds) {
        if (dxbcByResId.find(id) == dxbcByResId.end()) {
            LOGE("Missing resource id %u — is Lossless Scaling up to date?", id);
            return kErrMissingResource;
        }
    }

    int translated = 0;
    for (const auto &[resId, dxbc] : dxbcByResId) {
        // We only translate the ones we know belong to LSFG. Any other RCDATA
        // in the DLL (e.g. version strings) would fail DXBC parsing.
        const bool known =
            std::find(std::begin(kResourceIds), std::end(kResourceIds), resId) != std::end(kResourceIds);
        if (!known) {
            continue;
        }

        std::vector<uint8_t> spirv;
        try {
            spirv = translate_dxbc_to_spirv(dxbc);
        } catch (const std::exception &e) {
            LOGE("DXBC→SPIR-V failed for resource %u: %s", resId, e.what());
            return kErrTranslationFailed;
        }
        if (spirv.empty()) {
            LOGE("Empty SPIR-V for resource %u", resId);
            return kErrTranslationFailed;
        }

        char path[512];
        std::snprintf(path, sizeof(path), "%s/%u.spv", cacheDir.c_str(), resId);
        if (!write_file(path, spirv)) {
            LOGE("Failed to write %s", path);
            return kErrWriteFailed;
        }
        ++translated;
    }

    LOGI("Translated %d shaders into %s", translated, cacheDir.c_str());
    return kOk;
}

std::vector<uint8_t> load_cached_spirv(const std::string &cacheDir, uint32_t resId) {
    char path[512];
    std::snprintf(path, sizeof(path), "%s/%u.spv", cacheDir.c_str(), resId);
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return {};
    }
    const auto size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(static_cast<size_t>(size));
    f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(size));
    cap_spirv_version(out);  // fix shaders cached before the SPIR-V 1.3 downgrade
    return out;
}

// Mirror of Extract::nameIdxTable in lsfg-vk-android/src/extract/extract.cpp
// (Steam Deck side). Framegen asks for shaders by symbolic name; on Android
// we cache them on disk by numeric resource ID, so we need to translate.
uint32_t shader_name_to_resource_id(const std::string &name) {
    static const std::unordered_map<std::string, uint32_t> kTable = {
        { "mipmaps",     255 },
        { "alpha[0]",    267 },
        { "alpha[1]",    268 },
        { "alpha[2]",    269 },
        { "alpha[3]",    270 },
        { "beta[0]",     275 },
        { "beta[1]",     276 },
        { "beta[2]",     277 },
        { "beta[3]",     278 },
        { "beta[4]",     279 },
        { "gamma[0]",    257 },
        { "gamma[1]",    259 },
        { "gamma[2]",    260 },
        { "gamma[3]",    261 },
        { "gamma[4]",    262 },
        { "delta[0]",    257 },
        { "delta[1]",    263 },
        { "delta[2]",    264 },
        { "delta[3]",    265 },
        { "delta[4]",    266 },
        { "delta[5]",    258 },
        { "delta[6]",    271 },
        { "delta[7]",    272 },
        { "delta[8]",    273 },
        { "delta[9]",    274 },
        { "generate",    256 },
        { "p_mipmaps",   255 },
        { "p_alpha[0]",  290 },
        { "p_alpha[1]",  291 },
        { "p_alpha[2]",  292 },
        { "p_alpha[3]",  293 },
        { "p_beta[0]",   298 },
        { "p_beta[1]",   299 },
        { "p_beta[2]",   300 },
        { "p_beta[3]",   301 },
        { "p_beta[4]",   302 },
        { "p_gamma[0]",  280 },
        { "p_gamma[1]",  282 },
        { "p_gamma[2]",  283 },
        { "p_gamma[3]",  284 },
        { "p_gamma[4]",  285 },
        { "p_delta[0]",  280 },
        { "p_delta[1]",  286 },
        { "p_delta[2]",  287 },
        { "p_delta[3]",  288 },
        { "p_delta[4]",  289 },
        { "p_delta[5]",  281 },
        { "p_delta[6]",  294 },
        { "p_delta[7]",  295 },
        { "p_delta[8]",  296 },
        { "p_delta[9]",  297 },
        { "p_generate",  256 },
    };
    auto it = kTable.find(name);
    return it == kTable.end() ? 0u : it->second;
}

} // namespace lsfg_android
