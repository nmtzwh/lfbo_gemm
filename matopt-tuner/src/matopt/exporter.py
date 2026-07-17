from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .protocol import Workload, canonical_json, stable_hash
from .runner import MatOptRunner

AOT_SCHEMA = "aot_bundle_v1"
OBJECTIVES = {"one_shot", "steady", "throughput"}


class ExportError(RuntimeError):
    pass


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExportError(f"{where} must be an object")
    return value


def load_result(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read tuning result: {exc}") from exc
    result = _object(value, "result")
    if result.get("schema_version") != 1:
        raise ExportError("unsupported tuning result schema")
    return result


def selected_record(result: Mapping[str, Any], objective: str) -> dict[str, Any]:
    if objective not in OBJECTIVES:
        raise ExportError(f"invalid objective: {objective}")
    selected = _object(result.get("selected"), "result.selected")
    record = _object(selected.get(objective), f"result.selected.{objective}")
    if record.get("state") != "benchmarked":
        raise ExportError("selected record was not benchmarked")
    response = _object(record.get("response"), "selected response")
    measured = _object(response.get("measurement"), "selected measurement")
    if measured.get("correct") is not True or measured.get("stable") is not True:
        raise ExportError("selected record must be correct and stable")
    _object(record.get("plan"), "selected plan")
    _object(response.get("finalized"), "selected finalized plan")
    return record


def _workload(result: Mapping[str, Any]) -> Workload:
    value = _object(result.get("workload"), "result.workload")
    try:
        workload = Workload(**value)
        workload.validate()
    except (TypeError, ValueError) as exc:
        raise ExportError(f"invalid workload: {exc}") from exc
    return workload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_artifact(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ExportError(f"missing {label} artifact")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ExportError(f"{label} artifact escapes capture directory") from exc
    if not candidate.is_file():
        raise ExportError(f"missing {label} artifact: {relative}")
    return candidate


def validate_bundle(capture: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    bundle = _object(capture.get("aot_bundle"), "capture.aot_bundle")
    if bundle.get("schema") != AOT_SCHEMA:
        raise ExportError("native runner does not implement aot_bundle_v1")
    images = bundle.get("images")
    if not isinstance(images, list) or not images:
        raise ExportError("AOT bundle contains no code images")
    checked: list[dict[str, Any]] = []
    expected_ordinal: dict[str, int] = {}
    for index, raw in enumerate(images):
        image = _object(raw, f"image[{index}]")
        group = image.get("group")
        ordinal = image.get("ordinal")
        if not isinstance(group, str) or not group:
            raise ExportError(f"image[{index}] has no group")
        if ordinal != expected_ordinal.get(group, 0):
            raise ExportError(f"non-contiguous image ordinal in group {group}")
        expected_ordinal[group] = int(ordinal) + 1
        path = _safe_artifact(root, image.get("file"), f"image[{index}]")
        actual_size = path.stat().st_size
        actual_hash = _sha256(path)
        if image.get("size") != actual_size or image.get("sha256") != actual_hash:
            raise ExportError(f"code image integrity failure: {path.name}")
        if not isinstance(image.get("alignment"), int) or image["alignment"] <= 0:
            raise ExportError(f"invalid code image alignment: {path.name}")
        checked.append(dict(image))
    return checked


def kernel_id(manifest_without_id: Mapping[str, Any]) -> str:
    return stable_hash(manifest_without_id)[:24]


def _cpp_string(value: Any) -> str:
    return json.dumps(str(value))


def generated_header(kernel: str, manifest: Mapping[str, Any]) -> str:
    w = manifest["workload"]
    identity = manifest["tuning_identity"]
    bundle = manifest["aot_bundle"]
    packed = int(manifest["selected_measurement"].get("packed_bytes", 0))
    scratch = int(manifest["finalized_plan"].get("scratchpad_bytes", 0))
    requested_json = canonical_json(manifest["requested_plan"])
    finalized_json = canonical_json(manifest["finalized_plan"])
    manifest_json = canonical_json(manifest)
    symbol = f"matopt_{kernel}"
    namespace = f"k_{kernel}"
    comment = f"""/*
 * Fixed MatOpt AOT kernel {kernel}
 * Shape: M={w['m']} N={w['n']} K={w['k']}; FP32 row-major;
 * lda=K, ldb=N, ldc=N; alpha=1, beta=0.
 * Threads: exactly {w['threads']}; pin the process to exactly that many CPUs
 * before OpenMP initializes. ISA: {bundle['architecture']} {bundle['isa']};
 * vector width: {bundle['vector_bits']} bits.
 * Unsupported: transpose, batch, arbitrary strides, post-ops, aliasing,
 * reentrancy, concurrent calls, and any other shape.
 * Objective: {manifest['objective']}; packing: {manifest['packing_lifecycle']}.
 * Requested plan: {requested_json}.
 * Finalized plan: {finalized_json}.
 * Scratchpad: {scratch} bytes; packed weights: {packed} bytes.
 * Tuning identity: {identity.get('description', 'unknown')}.
 * Dependencies: {', '.join(manifest.get('dynamic_dependencies', []))}.
 * All kernel machine code is AOT-embedded. Construction performs no JIT.
 */"""
    return f"""#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

{comment}

extern \"C\" {{
struct {symbol}_runtime_info {{ int compatible; int warned; std::uint64_t jit_events; }};
void* {symbol}_create();
void {symbol}_destroy(void*) noexcept;
int {symbol}_prepare(void*, const float*) noexcept;
int {symbol}_run(void*, const float*, const float*, float*) noexcept;
int {symbol}_run_prepared(void*, const float*, float*) noexcept;
{symbol}_runtime_info {symbol}_info(const void*) noexcept;
}}

namespace matopt::{namespace} {{
struct KernelConstraints {{
    std::int64_t m, n, k, lda, ldb, ldc;
    float alpha, beta;
    int threads, vector_bits;
    std::size_t scratchpad_bytes, packed_weight_bytes;
    const char* dtype;
    const char* layout;
    const char* architecture;
    const char* isa;
    const char* objective;
    const char* packing_lifecycle;
    const char* tuning_fingerprint;
}};
inline constexpr KernelConstraints constraints {{{w['m']}, {w['n']}, {w['k']},
        {w['k']}, {w['n']}, {w['n']}, 1.0f, 0.0f,
        {w['threads']}, {bundle['vector_bits']}, {scratch}, {packed},
        \"f32\", \"dense_row_major\", {_cpp_string(bundle['architecture'])},
        {_cpp_string(bundle['isa'])}, {_cpp_string(manifest['objective'])},
        {_cpp_string(manifest['packing_lifecycle'])},
        {_cpp_string(identity.get('fingerprint', 'unknown'))}}};
inline constexpr const char manifest_json[] = {_cpp_string(manifest_json)};

struct RuntimeInfo {{ bool compatible; bool warned; std::uint64_t jit_events; }};

class MatMul {{
public:
    MatMul() : handle_({symbol}_create()) {{
        if (!handle_) throw std::runtime_error("MatOpt AOT construction failed");
    }}
    MatMul(const MatMul&) = delete;
    MatMul& operator=(const MatMul&) = delete;
    MatMul(MatMul&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {{}}
    MatMul& operator=(MatMul&& other) noexcept {{
        if (this != &other) {{ {symbol}_destroy(handle_); handle_ = std::exchange(other.handle_, nullptr); }}
        return *this;
    }}
    ~MatMul() {{ {symbol}_destroy(handle_); }}
    void prepare_weights(const float* b) {{ check({symbol}_prepare(handle_, b)); }}
    void run(const float* a, const float* b, float* c) {{ check({symbol}_run(handle_, a, b, c)); }}
    void run_prepared(const float* a, float* c) {{ check({symbol}_run_prepared(handle_, a, c)); }}
    RuntimeInfo runtime_info() const noexcept {{
        auto i = {symbol}_info(handle_); return {{i.compatible != 0, i.warned != 0, i.jit_events}};
    }}
private:
    static void check(int status) {{ if (status) throw std::runtime_error("MatOpt AOT call failed"); }}
    void* handle_ = nullptr;
}};
}}
"""


def _cmake_config(kernel: str) -> str:
    return f"""include(CMakeFindDependencyMacro)
get_filename_component(_matopt_prefix "${{CMAKE_CURRENT_LIST_DIR}}/../../.." ABSOLUTE)
add_library(MatOptKernel::{kernel} SHARED IMPORTED)
set_target_properties(MatOptKernel::{kernel} PROPERTIES
  IMPORTED_LOCATION "${{_matopt_prefix}}/lib/libmatopt_kernel_{kernel}.so"
  INTERFACE_INCLUDE_DIRECTORIES "${{_matopt_prefix}}/include")
"""


def export_package(
    *,
    result_path: str | os.PathLike[str],
    objective: str,
    runner_path: str | os.PathLike[str],
    onednn_build: str | os.PathLike[str],
    build_dir: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    result = load_result(result_path)
    record = selected_record(result, objective)
    workload = _workload(result)
    runner = MatOptRunner(runner_path)
    capabilities = runner.capabilities(workload)
    if capabilities.get("status") != "capabilities":
        raise ExportError(f"live runner capability discovery failed: {capabilities}")
    expected = result.get("runner_fingerprint")
    live = capabilities.get("fingerprint")
    if not isinstance(expected, str) or expected != live:
        raise ExportError("runner fingerprint does not match the tuning result")
    plan = record["plan"]
    inspected = runner.inspect(workload, plan, live)
    if inspected.get("status") != "accepted":
        raise ExportError(f"selected plan no longer finalizes: {inspected}")
    saved_finalized = record["response"]["finalized"]
    if inspected.get("finalized") != saved_finalized:
        raise ExportError("re-finalized descriptor or realized microtile changed")
    onednn = Path(onednn_build).resolve()
    if not onednn.is_dir():
        raise ExportError(f"oneDNN build directory does not exist: {onednn}")
    build = Path(build_dir).resolve()
    build.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="capture-", dir=build) as temporary:
        capture_root = Path(temporary)
        capture = runner.capture_aot(workload, plan, live, capture_root)
        if capture.get("status") != "captured":
            raise ExportError(f"native AOT capture failed: {capture}")
        if capture.get("finalized") != saved_finalized:
            raise ExportError("capture finalized a different descriptor bundle")
        images = validate_bundle(capture, capture_root)
        aot = dict(capture["aot_bundle"])
        aot["images"] = images
        build_identity = dict(capture.get("build", {}))
        builder_path = Path(__file__).with_name("aot_builder.py")
        build_identity["exporter_builder_sha256"] = _sha256(builder_path)
        manifest_base: dict[str, Any] = {
            "schema_version": 1,
            "aot_schema": AOT_SCHEMA,
            "workload": workload.to_dict(),
            "objective": objective,
            "selected_measurement": record["response"]["measurement"],
            "requested_plan": plan,
            "finalized_plan": saved_finalized,
            "packing_lifecycle": plan.get("pack_b", "direct"),
            "tuning_identity": {
                "fingerprint": live,
                "description": capabilities.get("identity", "unknown"),
                "original_cpu_mask": workload.cpus,
            },
            "aot_bundle": aot,
            "build": build_identity,
            "dynamic_dependencies": capture.get("dynamic_dependencies", []),
        }
        library_value = capture.get("library")
        if library_value is None:
            from .aot_builder import build_shared_library, dynamic_dependencies

            source = Path(__file__).resolve().parents[3] / "oneDNN"
            probe = build_shared_library(
                kernel_id="probe",
                images=images,
                capture_root=capture_root,
                workload=workload.to_dict(),
                plan=plan,
                aot_bundle=aot,
                tuning_identity=str(capabilities.get("identity", "unknown")),
                onednn_source=source,
                onednn_build=onednn,
                build_dir=build,
            )
            manifest_base["dynamic_dependencies"] = dynamic_dependencies(probe)
        kid = kernel_id(manifest_base)
        manifest = dict(manifest_base, kernel_id=kid)
        if library_value is None:
            library = build_shared_library(
                kernel_id=kid,
                images=images,
                capture_root=capture_root,
                workload=workload.to_dict(),
                plan=plan,
                aot_bundle=aot,
                tuning_identity=str(capabilities.get("identity", "unknown")),
                onednn_source=source,
                onednn_build=onednn,
                build_dir=build,
            )
        else:
            library = _safe_artifact(
                capture_root, library_value, "shared library"
            )
        destination = Path(output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            (stage / "include").mkdir()
            (stage / "lib/cmake/MatOptKernel").mkdir(parents=True)
            (stage / "share/matopt").mkdir(parents=True)
            (stage / "share/licenses").mkdir(parents=True)
            (stage / "include" / f"matopt_kernel_{kid}.hpp").write_text(
                generated_header(kid, manifest), encoding="utf-8"
            )
            shutil.copy2(library, stage / "lib" / f"libmatopt_kernel_{kid}.so")
            (stage / "lib/cmake/MatOptKernel/MatOptKernelConfig.cmake").write_text(
                _cmake_config(kid), encoding="utf-8"
            )
            (stage / "share/matopt/manifest.json").write_text(
                canonical_json(manifest) + "\n", encoding="utf-8"
            )
            license_path = Path(__file__).resolve().parents[3] / "oneDNN/LICENSE"
            if not license_path.is_file():
                raise ExportError("oneDNN license is unavailable")
            shutil.copy2(license_path, stage / "share/licenses/oneDNN-LICENSE")
            validated = runner.validate_aot(workload, live, stage)
            if validated.get("status") != "validated" or validated.get("jit_events") != 0:
                raise ExportError(f"fresh-process AOT validation failed: {validated}")
            if destination.exists():
                raise ExportError(f"output already exists: {destination}")
            os.replace(stage, destination)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return {"output": str(destination), "kernel_id": kid, "manifest": manifest}
