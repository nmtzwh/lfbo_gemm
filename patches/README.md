# oneDNN patch

`onednn-v3.12.2-matopt.patch` is generated against the unmodified oneDNN
v3.12.2 tag at commit `76ef43d856388d95c1e23573a7d057766e5ee00e`.

Verify the base and patch before applying it:

```bash
git rev-parse HEAD
git apply --check /path/to/onednn-v3.12.2-matopt.patch
```

The expected `HEAD` is
`76ef43d856388d95c1e23573a7d057766e5ee00e`.

The patch includes the opt-in MatOpt tuner backend plus `aot_bundle_v1`
capture/replay hooks in the x64 and AArch64 JIT generator bases. The
`capture-aot` runner command verifies copied-address replay before the Python
exporter embeds images into ELF executable text; `validate-aot` then checks the
finished package in a fresh process without permitting JIT fallback.

It also includes the internal `perf-diag` runner command and address-free
worker-level BRGEMM trace hooks used by `matopt-tuner diagnose`. These remain
inside `DNNL_BUILD_MATOPT`; the public oneDNN ABI and ordinary builds are
unchanged.
