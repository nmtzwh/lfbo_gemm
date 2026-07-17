#include <benchmark/benchmark.h>
#include <cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernel_bridge.hpp"

namespace {
using clock_type = std::chrono::steady_clock;
namespace kernel = matopt_benchmark_kernel;
constexpr std::size_t M = kernel::constraints.m;
constexpr std::size_t N = kernel::constraints.n;
constexpr std::size_t K = kernel::constraints.k;

struct Data {
    std::vector<float> a = std::vector<float>(M * K);
    std::vector<float> b = std::vector<float>(K * N);
    std::vector<float> b2 = std::vector<float>(K * N);
    std::vector<float> c = std::vector<float>(M * N);
    std::vector<float> ref = std::vector<float>(M * N);
    Data() {
        for (std::size_t i = 0; i < a.size(); ++i) a[i] = float(int(i % 7) - 3);
        for (std::size_t i = 0; i < b.size(); ++i) b[i] = float(int(i % 5) - 2);
        for (std::size_t i = 0; i < b2.size(); ++i) b2[i] = float(int(i % 11) - 5);
        for (std::size_t i = 0; i < a.size(); i += 4093) a[i] += .125f;
        for (std::size_t i = 0; i < b.size(); i += 4099) b[i] -= .25f;
        for (std::size_t i = 0; i < b2.size(); i += 4099) b2[i] += .375f;
    }
};
Data data;
std::unique_ptr<kernel::MatMul> prepared;

void reference(const float *b) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.f,
            data.a.data(), K, b, N, 0.f, data.ref.data(), N);
}

void check(const char *mode) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < data.c.size(); ++i) {
        const float tolerance = 2e-4f * std::max(1.f, std::fabs(data.ref[i]));
        if (!std::isfinite(data.c[i]) || std::fabs(data.c[i] - data.ref[i]) > tolerance)
            if (++bad == 8) break;
    }
    if (bad) throw std::runtime_error(std::string(mode) + " full-matrix mismatch");
    const std::size_t samples[] {0, N - 1, (M / 2) * N + N / 2, M * N - 1};
    for (std::size_t index : samples) {
        const std::size_t row = index / N, col = index % N;
        double dot = 0, magnitude = 0;
        for (std::size_t k = 0; k < K; ++k) {
            const double product = double(data.a[row * K + k]) * data.b[k * N + col];
            dot += product; magnitude += std::fabs(product);
        }
        const double bound = 1e-6 + 8 * std::numeric_limits<float>::epsilon() * magnitude;
        if (std::fabs(double(data.c[index]) - dot) > bound)
            throw std::runtime_error(std::string(mode) + " FP64 sample mismatch");
    }
}

void preflight(const std::string &path) {
    kernel::MatMul op;
    reference(data.b.data());
    std::fill(data.c.begin(), data.c.end(), std::numeric_limits<float>::quiet_NaN());
    op.run(data.a.data(), data.b.data(), data.c.data()); check("one_shot");
    std::fill(data.c.begin(), data.c.end(), std::numeric_limits<float>::quiet_NaN());
    op.prepare_weights(data.b.data()); op.run_prepared(data.a.data(), data.c.data());
    check("prepared");
    reference(data.b2.data());
    op.prepare_weights(data.b2.data()); op.run_prepared(data.a.data(), data.c.data());
    std::swap(data.b, data.b2); check("reprepared"); std::swap(data.b, data.b2);
    const auto info = op.runtime_info();
    if (!info.compatible || info.jit_events != 0) throw std::runtime_error("AOT runtime preflight failed");
    std::ofstream out(path);
    out << "{\"schema_version\":1,\"correct\":true,\"jit_events\":"
        << info.jit_events << ",\"runtime_warning\":" << (info.warned ? "true" : "false")
        << ",\"openblas_configuration\":\"" << openblas_get_config() << "\"}\n";
}

template <class F> void manual(benchmark::State &state, F &&fn) {
    for (int i = 0; i < 3; ++i) fn();
    for (auto _ : state) {
        const auto begin = clock_type::now(); fn(); const auto end = clock_type::now();
        state.SetIterationTime(std::chrono::duration<double>(end - begin).count());
    }
    state.SetItemsProcessed(state.iterations());
}

void create(benchmark::State &state) { manual(state, [] { kernel::MatMul op; benchmark::DoNotOptimize(op.runtime_info()); }); }
void prepare_weights(benchmark::State &state) {
    kernel::MatMul op; manual(state, [&] { op.prepare_weights(data.b.data()); });
}
void one_shot(benchmark::State &state) {
    kernel::MatMul op; manual(state, [&] { op.run(data.a.data(), data.b.data(), data.c.data()); benchmark::ClobberMemory(); });
}
void steady(benchmark::State &state) {
    kernel::MatMul op; op.prepare_weights(data.b.data());
    manual(state, [&] { op.run_prepared(data.a.data(), data.c.data()); benchmark::ClobberMemory(); });
    state.counters["GFLOP/s"] = benchmark::Counter(2.0 * M * N * K, benchmark::Counter::kIsRate);
}
void openblas(benchmark::State &state) {
    manual(state, [] { cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.f,
        data.a.data(), K, data.b.data(), N, 0.f, data.c.data(), N); benchmark::ClobberMemory(); });
    state.counters["GFLOP/s"] = benchmark::Counter(2.0 * M * N * K, benchmark::Counter::kIsRate);
}
BENCHMARK(create)->Name("MatOpt/create")->UseManualTime();
BENCHMARK(prepare_weights)->Name("MatOpt/prepare_weights")->UseManualTime();
BENCHMARK(one_shot)->Name("MatOpt/one_shot")->UseManualTime();
BENCHMARK(steady)->Name("MatOpt/steady_throughput")->UseManualTime();
BENCHMARK(openblas)->Name("OpenBLAS/sgemm")->UseManualTime();
}

int main(int argc, char **argv) {
    std::string correctness;
    std::vector<char *> benchmark_args {argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind("--correctness=", 0) == 0) correctness = arg.substr(14);
        else benchmark_args.push_back(argv[i]);
    }
    if (correctness.empty()) throw std::runtime_error("--correctness is required");
    openblas_set_num_threads(kernel::constraints.threads);
    preflight(correctness);
    int benchmark_argc = static_cast<int>(benchmark_args.size());
    benchmark::Initialize(&benchmark_argc, benchmark_args.data());
    if (benchmark::ReportUnrecognizedArguments(benchmark_argc, benchmark_args.data())) return 2;
    benchmark::RunSpecifiedBenchmarks(); benchmark::Shutdown(); return 0;
}
