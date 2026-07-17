import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    os.environ.get("MATOPT_AOT_PACKAGE"), "MATOPT_AOT_PACKAGE is not set"
)
class NativeAOTPackageTests(unittest.TestCase):
    def test_header_only_consumer_links_and_runs(self):
        package = Path(os.environ["MATOPT_AOT_PACKAGE"]).resolve()
        manifest = json.loads(
            (package / "share/matopt/manifest.json").read_text(encoding="utf-8")
        )
        kid = manifest["kernel_id"]
        workload = manifest["workload"]
        source_text = f'''#include <cmath>
#include <vector>
#include "matopt_kernel_{kid}.hpp"
namespace kernel = matopt::k_{kid};
int main() {{
    constexpr size_t M={workload['m']}, N={workload['n']}, K={workload['k']};
    std::vector<float> a(M*K), b(K*N), c(M*N), first(M*N);
    for (size_t i=0;i<a.size();++i) a[i]=float(int(i%7)-3);
    for (size_t i=0;i<b.size();++i) b[i]=float(int(i%5)-2);
    kernel::MatMul op;
    op.run(a.data(), b.data(), c.data());
    first=c;
    op.prepare_weights(b.data());
    op.run_prepared(a.data(), c.data());
    for (size_t i=0;i<c.size();++i) if (std::fabs(c[i]-first[i])>1e-4f) return 2;
    for (float &value:b) value+=1.f;
    op.prepare_weights(b.data());
    op.run_prepared(a.data(), c.data());
    bool changed=false;
    for (size_t i=0;i<c.size();++i) changed|=std::fabs(c[i]-first[i])>1e-4f;
    const auto info=op.runtime_info();
    return changed && info.compatible && info.jit_events==0 ? 0 : 3;
}}
'''
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "consumer.cpp"
            executable = directory / "consumer"
            source.write_text(source_text, encoding="utf-8")
            compile_result = subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    str(source),
                    f"-I{package / 'include'}",
                    f"-L{package / 'lib'}",
                    f"-lmatopt_kernel_{kid}",
                    f"-Wl,-rpath,{package / 'lib'}",
                    "-o",
                    str(executable),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [
                    "taskset",
                    "-c",
                    workload["cpus"],
                    str(executable),
                ],
                env={**os.environ, "OMP_NUM_THREADS": str(workload["threads"])},
                text=True,
                capture_output=True,
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)


if __name__ == "__main__":
    unittest.main()
