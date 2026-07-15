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

