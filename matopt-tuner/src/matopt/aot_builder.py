from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .exporter import ExportError


def _symbol(kernel_id: str, index: int) -> str:
    return f"matopt_{kernel_id}_image_{index}"


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise ExportError(
            f"AOT build command failed ({command[0]}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout


def _assembly(
    kernel_id: str, images: list[Mapping[str, Any]], capture_root: Path
) -> str:
    lines = ['.section .text.matopt_aot,"ax",@progbits']
    for index, image in enumerate(images):
        symbol = _symbol(kernel_id, index)
        path = (capture_root / str(image["file"])).resolve()
        lines.extend(
            [
                f".balign {int(image['alignment'])}",
                f".globl {symbol}",
                f".hidden {symbol}",
                f".type {symbol}, @function",
                f"{symbol}:",
                f'.incbin "{str(path).replace(chr(34), chr(92) + chr(34))}"',
                f".size {symbol}, .-{symbol}",
            ]
        )
    lines.append('.section .note.GNU-stack,"",@progbits')
    return "\n".join(lines) + "\n"


def _plan_assignments(plan: Mapping[str, Any]) -> str:
    integer_fields = (
        "version",
        "M_blk",
        "N_blk",
        "K_blk",
        "M_chunk_size",
        "N_chunk_size",
        "K_chunk_size",
        "brgemm_batch_size",
        "nthr_k",
        "bd_block",
        "ld_block2",
    )
    lines = ["matopt::plan_t result;"]
    for field in integer_fields:
        if field in plan:
            lines.append(f"result.{field} = {int(plan[field])};")
    pack_a = {
        "direct": "direct",
        "per_call_padded": "per_call_padded",
    }[str(plan["pack_a"])]
    pack_b = {
        "direct": "direct",
        "per_call_n32": "per_call_n32",
        "per_call_n64": "per_call_n64",
        "persistent_n32": "persistent_n32",
        "persistent_n64": "persistent_n64",
    }[str(plan["pack_b"])]
    loop = {
        "default": "default_order",
        "one-load": "one_load",
    }[str(plan["loop_order"])]
    lines.extend(
        [
            f"result.pack_a = matopt::pack_a_t::{pack_a};",
            f"result.pack_b = matopt::pack_b_t::{pack_b};",
            f"result.loop_order = matopt::loop_order_t::{loop};",
            "return result;",
        ]
    )
    return "\n    ".join(lines)


def _wrapper_source(
    kernel_id: str,
    images: list[Mapping[str, Any]],
    workload: Mapping[str, Any],
    plan: Mapping[str, Any],
    aot_bundle: Mapping[str, Any],
    tuning_identity: str,
) -> str:
    prefix = f"matopt_{kernel_id}"
    declarations = "\n".join(
        f'extern "C" const unsigned char {_symbol(kernel_id, i)}[];'
        for i in range(len(images))
    )
    entries = ",\n".join(
        "    {"
        + json.dumps(str(image["group"]))
        + f", {int(image['ordinal'])}, "
        + json.dumps(str(image["name"]))
        + f", {_symbol(kernel_id, i)}, {int(image['size'])}, "
        + f"{int(image['alignment'])}" + "}"
        for i, image in enumerate(images)
    )
    persistent = str(plan["pack_b"]).startswith("persistent_")
    identity_fields = {
        part.split("=", 1)[0]: part.split("=", 1)[1]
        for part in tuning_identity.split(";")[1:]
        if "=" in part
    }
    cache_identity = "|".join(
        identity_fields.get(name, "unknown") for name in ("L1", "L2", "L3")
    )
    expected_hashes = ", ".join(
        json.dumps(str(image["sha256"])) for image in images
    )
    return f'''#include <atomic>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <new>
#include <sstream>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#if defined(__aarch64__)
#include <linux/prctl.h>
#include <sys/prctl.h>
#endif

#include <omp.h>
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "cpu/matmul/matopt_backend.hpp"

namespace matopt = dnnl::impl::cpu::matopt;
{declarations}

namespace {{
constexpr int64_t M = {int(workload['m'])};
constexpr int64_t N = {int(workload['n'])};
constexpr int64_t K = {int(workload['k'])};
constexpr int threads = {int(workload['threads'])};
constexpr bool persistent_b = {'true' if persistent else 'false'};
constexpr const char *required_architecture = {json.dumps(str(aot_bundle['architecture']))};
constexpr const char *required_isa = {json.dumps(str(aot_bundle['isa']))};
constexpr int required_vector_bits = {int(aot_bundle['vector_bits'])};
constexpr const char *tuning_cpu_identity = {json.dumps(tuning_identity.split(';', 1)[0])};
constexpr const char *tuning_cache_identity = {json.dumps(cache_identity)};

const matopt::aot_replay_image_t images[] = {{
{entries}
}};
const char *expected_hashes[] = {{{expected_hashes}}};

uint32_t rotate_right(uint32_t value, unsigned shift) {{
    return (value >> shift) | (value << (32 - shift));
}}

std::string sha256(const uint8_t *data, size_t size) {{
    static constexpr uint32_t constants[64] = {{
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2}};
    uint32_t h[8] = {{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
            0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    const uint64_t bit_length = static_cast<uint64_t>(size) * 8;
    const size_t padded = ((size + 9 + 63) / 64) * 64;
    std::vector<uint8_t> message(padded, 0);
    std::memcpy(message.data(), data, size);
    message[size] = 0x80;
    for (unsigned i = 0; i < 8; ++i)
        message[padded - 1 - i] = static_cast<uint8_t>(bit_length >> (i * 8));
    for (size_t offset = 0; offset < padded; offset += 64) {{
        uint32_t w[64];
        for (unsigned i = 0; i < 16; ++i) {{
            const auto *p = message.data() + offset + i * 4;
            w[i] = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16)
                    | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
        }}
        for (unsigned i = 16; i < 64; ++i) {{
            const uint32_t s0 = rotate_right(w[i - 15], 7)
                    ^ rotate_right(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotate_right(w[i - 2], 17)
                    ^ rotate_right(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }}
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (unsigned i = 0; i < 64; ++i) {{
            const uint32_t s1=rotate_right(e,6)^rotate_right(e,11)^rotate_right(e,25);
            const uint32_t choice=(e&f)^((~e)&g);
            const uint32_t t1=hh+s1+choice+constants[i]+w[i];
            const uint32_t s0=rotate_right(a,2)^rotate_right(a,13)^rotate_right(a,22);
            const uint32_t majority=(a&b)^(a&c)^(b&c);
            const uint32_t t2=s0+majority;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }}
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }}
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (unsigned i = 0; i < 32; ++i) {{
        const uint8_t byte = static_cast<uint8_t>(h[i / 4] >> (24 - 8 * (i % 4)));
        result[2 * i] = hex[byte >> 4]; result[2 * i + 1] = hex[byte & 15];
    }}
    return result;
}}

matopt::plan_t selected_plan() {{
    {_plan_assignments(plan)}
}}

bool validate_process() {{
#if !defined(__linux__)
    throw std::runtime_error("MatOpt AOT requires Linux");
#endif
#if defined(__x86_64__)
    if (std::string(required_architecture) != "x86_64")
        throw std::runtime_error("MatOpt AOT architecture mismatch");
#elif defined(__aarch64__)
    if (std::string(required_architecture) != "aarch64")
        throw std::runtime_error("MatOpt AOT architecture mismatch");
    if ((prctl(PR_SVE_GET_VL) & PR_SVE_VL_LEN_MASK)
            != required_vector_bits / 8)
        throw std::runtime_error("MatOpt AOT SVE vector-length mismatch");
#else
    throw std::runtime_error("unsupported MatOpt AOT architecture");
#endif
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0
            || CPU_COUNT(&affinity) != threads)
        throw std::runtime_error("MatOpt AOT affinity cardinality mismatch");
    if (omp_get_max_threads() != threads)
        throw std::runtime_error("MatOpt AOT OpenMP thread-count mismatch");
    const std::string effective = dnnl_cpu_isa2str(dnnl_get_effective_cpu_isa());
    const bool isa_ok = effective.find(required_isa) != std::string::npos
            || (std::string(required_isa).find("AVX2") != std::string::npos
                    && effective.find("AVX-512") != std::string::npos);
    if (!isa_ok)
        throw std::runtime_error("MatOpt AOT ISA mismatch: " + effective);
    for (size_t i = 0; i < sizeof(images) / sizeof(images[0]); ++i)
        if (sha256(images[i].code, images[i].size) != expected_hashes[i])
            throw std::runtime_error("MatOpt AOT image-integrity mismatch");
    bool warned = true;
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line))
        if (line.find("model name") != std::string::npos
                || line.find("CPU part") != std::string::npos)
            warned = line != tuning_cpu_identity;
    auto cache_size = [](const char *path) {{
        std::ifstream input(path);
        std::string value;
        return std::getline(input, value) ? value : "unknown";
    }};
    const std::string cache = cache_size(
            "/sys/devices/system/cpu/cpu0/cache/index0/size") + "|"
            + cache_size("/sys/devices/system/cpu/cpu0/cache/index2/size")
            + "|" + cache_size(
                    "/sys/devices/system/cpu/cpu0/cache/index3/size");
    return warned || cache != tuning_cache_identity;
}}

struct state_t {{
    dnnl::engine engine {{dnnl::engine::kind::cpu, 0}};
    dnnl::memory::desc a_md {{{{M, K}}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab}};
    dnnl::memory::desc b_md {{{{K, N}}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab}};
    dnnl::memory::desc c_md {{{{M, N}}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab}};
    dnnl::matmul::primitive_desc pd;
    dnnl::matmul primitive;
    dnnl::memory a, b_plain, b_exec, b_prepared, c, scratchpad;
    dnnl::reorder reorder_b;
    dnnl::stream stream {{engine}};
    bool prepared = false;
    bool warned = false;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;

    state_t() {{
        warned = validate_process();
        dnnl::set_primitive_cache_capacity(0);
        a = dnnl::memory(a_md, engine, DNNL_MEMORY_NONE);
        b_plain = dnnl::memory(b_md, engine, DNNL_MEMORY_NONE);
        c = dnnl::memory(c_md, engine, DNNL_MEMORY_NONE);
        matopt::aot_begin_replay(images, sizeof(images) / sizeof(images[0]));
        try {{
            const auto plan = selected_plan();
            matopt::scoped_plan_t plan_scope(&plan);
            matopt::aot_set_group("matmul");
            const auto requested_b = persistent_b
                    ? dnnl::memory::desc(b_md.get_dims(),
                            dnnl::memory::data_type::f32,
                            dnnl::memory::format_tag::any)
                    : b_md;
            dnnl::primitive_attr attr;
            attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
            pd = dnnl::matmul::primitive_desc(
                    engine, a_md, requested_b, c_md, attr);
            primitive = dnnl::matmul(pd);
            b_exec = persistent_b
                    ? dnnl::memory(pd.weights_desc(), engine)
                    : dnnl::memory(b_md, engine, DNNL_MEMORY_NONE);
            if (!persistent_b)
                b_prepared = dnnl::memory(b_md, engine);
            if (pd.scratchpad_desc().get_size())
                scratchpad = dnnl::memory(pd.scratchpad_desc(), engine);
            if (persistent_b) {{
                matopt::aot_set_group("reorder");
                reorder_b = dnnl::reorder(b_plain, b_exec);
            }}
            matopt::aot_set_group("matmul");
            std::string error;
            if (!matopt::aot_end_replay(error))
                throw std::runtime_error(error);
        }} catch (...) {{
            matopt::aot_abort();
            throw;
        }}
    }}

    struct call_guard_t {{
        std::atomic_flag &flag;
        explicit call_guard_t(std::atomic_flag &value) : flag(value) {{
            if (flag.test_and_set(std::memory_order_acquire))
                throw std::runtime_error("concurrent MatOpt AOT call");
        }}
        ~call_guard_t() {{ flag.clear(std::memory_order_release); }}
    }};

    void prepare(const float *b) {{
        if (!b) throw std::invalid_argument("null B");
        if (persistent_b) {{
            b_plain.set_data_handle(const_cast<float *>(b));
            reorder_b.execute(stream, b_plain, b_exec);
            stream.wait();
        }} else {{
            std::memcpy(b_prepared.get_data_handle(), b,
                    static_cast<size_t>(K * N) * sizeof(float));
        }}
        prepared = true;
    }}

    void execute(const float *a_ptr, const float *b_ptr, float *c_ptr,
            bool use_prepared) {{
        call_guard_t guard(busy);
        if (!a_ptr || !c_ptr) throw std::invalid_argument("null A or C");
        if (persistent_b) {{
            if (!use_prepared) prepare(b_ptr);
            if (!prepared) throw std::runtime_error("weights are not prepared");
        }} else {{
            if (use_prepared) {{
                if (!prepared)
                    throw std::runtime_error("weights are not prepared");
                b_exec.set_data_handle(b_prepared.get_data_handle());
            }} else {{
                if (!b_ptr) throw std::runtime_error("direct-B kernel requires B");
                b_exec.set_data_handle(const_cast<float *>(b_ptr));
            }}
        }}
        a.set_data_handle(const_cast<float *>(a_ptr));
        c.set_data_handle(c_ptr);
        std::unordered_map<int, dnnl::memory> args {{{{DNNL_ARG_SRC, a}},
                {{DNNL_ARG_WEIGHTS, b_exec}}, {{DNNL_ARG_DST, c}}}};
        if (pd.scratchpad_desc().get_size())
            args.emplace(DNNL_ARG_SCRATCHPAD, scratchpad);
        primitive.execute(stream, args);
        stream.wait();
    }}
}};
}} // namespace

extern "C" {{
struct {prefix}_runtime_info {{
    int compatible;
    int warned;
    std::uint64_t jit_events;
}};

__attribute__((visibility("default"))) void *{prefix}_create() noexcept {{
    try {{ return new state_t(); }} catch (...) {{ return nullptr; }}
}}
__attribute__((visibility("default"))) void {prefix}_destroy(void *p) noexcept {{
    delete static_cast<state_t *>(p);
}}
__attribute__((visibility("default"))) int {prefix}_prepare(
        void *p, const float *b) noexcept {{
    try {{
        auto *state = static_cast<state_t *>(p);
        if (!state) return 1;
        state_t::call_guard_t guard(state->busy);
        state->prepare(b);
        return 0;
    }} catch (...) {{ return 1; }}
}}
__attribute__((visibility("default"))) int {prefix}_run(
        void *p, const float *a, const float *b, float *c) noexcept {{
    try {{
        auto *state = static_cast<state_t *>(p);
        if (!state) return 1;
        state->execute(a, b, c, false);
        return 0;
    }} catch (...) {{ return 1; }}
}}
__attribute__((visibility("default"))) int {prefix}_run_prepared(
        void *p, const float *a, float *c) noexcept {{
    try {{
        auto *state = static_cast<state_t *>(p);
        if (!state) return 1;
        state->execute(a, nullptr, c, true);
        return 0;
    }} catch (...) {{ return 1; }}
}}
__attribute__((visibility("default"))) {prefix}_runtime_info {prefix}_info(
        const void *p) noexcept {{
    const auto *state = static_cast<const state_t *>(p);
    return {{state ? 1 : 0, state && state->warned ? 1 : 0, 0}};
}}
}}
'''


def _version_script(kernel_id: str) -> str:
    return (
        "MATOPT_1.0 {\n  global:\n    matopt_"
        + kernel_id
        + "_*;\n  local: *;\n};\n"
    )


def build_shared_library(
    *,
    kernel_id: str,
    images: list[Mapping[str, Any]],
    capture_root: Path,
    workload: Mapping[str, Any],
    plan: Mapping[str, Any],
    aot_bundle: Mapping[str, Any],
    tuning_identity: str,
    onednn_source: Path,
    onednn_build: Path,
    build_dir: Path,
) -> Path:
    output_dir = build_dir / kernel_id
    output_dir.mkdir(parents=True, exist_ok=True)
    assembly = output_dir / "images.S"
    wrapper = output_dir / "wrapper.cpp"
    version = output_dir / "exports.map"
    library = output_dir / f"libmatopt_kernel_{kernel_id}.so"
    assembly.write_text(_assembly(kernel_id, images, capture_root), encoding="utf-8")
    wrapper.write_text(
        _wrapper_source(
            kernel_id, images, workload, plan, aot_bundle, tuning_identity
        ),
        encoding="utf-8",
    )
    version.write_text(_version_script(kernel_id), encoding="utf-8")
    static_library = onednn_build / "src/libdnnl.a"
    if not static_library.is_file():
        raise ExportError(f"patched static oneDNN was not found: {static_library}")
    _run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-fvisibility=hidden",
            "-shared",
            str(wrapper),
            str(assembly),
            str(static_library),
            f"-I{onednn_source}",
            f"-I{onednn_source / 'src'}",
            f"-I{onednn_source / 'include'}",
            f"-I{onednn_build / 'include'}",
            "-fopenmp",
            "-ldl",
            "-lpthread",
            "-lm",
            f"-Wl,--version-script={version}",
            "-Wl,--exclude-libs,ALL",
            "-Wl,-z,relro,-z,now,-z,noexecstack",
            "-o",
            str(library),
        ]
    )
    return library


def dynamic_dependencies(library: Path) -> list[str]:
    output = _run(["readelf", "-d", str(library)])
    return sorted(set(re.findall(r"Shared library: \[(.+?)\]", output)))
