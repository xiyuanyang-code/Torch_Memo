import os
import torch
import argparse
import matplotlib.pyplot as plt
import torch.utils.benchmark as benchmark
from torch.cuda import synchronize

# restrict the numbers for running this program
# *Of course you can comment this if you are rich enough...
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def test_speed(iterations):
    """Test compute speed on CPU and GPU(s), measuring only the actual computation."""
    print("Starting performance test...\n")
    results = []

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Define matrix size (adjust if needed to fit memory)
    size = 5000
    print(f"Using matrix size: {size}x{size}\n")

    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")

        # Prepare CPU tensors
        a_cpu = torch.randn(size, size, dtype=torch.float32)
        b_cpu = torch.randn(size, size, dtype=torch.float32)

        # Benchmark CPU (only compute time)
        cpu_time = (
            benchmark.Timer(
                stmt="a_cpu + b_cpu", globals={"a_cpu": a_cpu, "b_cpu": b_cpu}
            )
            .timeit(10)
            .mean
            * 1000
        )  # in ms

        # Benchmark Single GPU (create data directly on GPU)
        a_gpu = torch.randn(size, size, dtype=torch.float32, device="cuda")
        b_gpu = torch.randn(size, size, dtype=torch.float32, device="cuda")

        # Warm-up
        for _ in range(5):
            c = a_gpu + b_gpu
        synchronize()

        # Measure with CUDA Events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(10):  # run multiple times for better averaging
            c_gpu_single = a_gpu + b_gpu
        end.record()
        synchronize()
        single_gpu_time = (start.elapsed_time(end)) / 10  # average per run

        multi_gpu_time = None
        if num_gpus > 1:
            a_chunks = list(a_gpu.chunk(num_gpus, dim=0))
            b_chunks = list(b_gpu.chunk(num_gpus, dim=0))

            # Move each chunk to its respective GPU
            for idx in range(num_gpus):
                with torch.cuda.device(idx):
                    a_chunks[idx] = a_chunks[idx].to(f"cuda:{idx}")
                    b_chunks[idx] = b_chunks[idx].to(f"cuda:{idx}")

            # Warm-up
            outputs = []
            for idx in range(num_gpus):
                with torch.cuda.device(idx):
                    outputs.append(a_chunks[idx] + b_chunks[idx])
            synchronize()

            # Timing
            start_event_multi = torch.cuda.Event(enable_timing=True)
            end_event_multi = torch.cuda.Event(enable_timing=True)

            start_event_multi.record()

            outputs = []
            for idx in range(num_gpus):
                with torch.cuda.device(idx):
                    outputs.append(a_chunks[idx] + b_chunks[idx])

            end_event_multi.record()
            synchronize()
            multi_gpu_time = start_event_multi.elapsed_time(end_event_multi)  # in ms

            outputs_cpu = [out.cpu() for out in outputs]
            c_multi_cpu = torch.cat(outputs_cpu, dim=0)

        # Compute max difference between CPU and GPU results
        diff_single = (a_gpu.cpu() + b_gpu.cpu() - (a_cpu + b_cpu)).abs().max().item()
        diff_multi = None
        if num_gpus > 1:
            diff_multi = (c_multi_cpu - (a_cpu + b_cpu)).abs().max().item()

        print(f"CPU Time: {cpu_time:.2f} ms")
        print(f"Single GPU Time: {single_gpu_time:.2f} ms")
        if multi_gpu_time is not None:
            print(f"Multi-GPU Time: {multi_gpu_time:.2f} ms")

        print(f"Max difference (CPU vs Single GPU): {diff_single:.6f}")
        if diff_multi is not None:
            print(f"Max difference (CPU vs Multi-GPU): {diff_multi:.6f}")

        results.append(
            {
                "cpu_time": cpu_time,
                "single_gpu_time": single_gpu_time,
                "multi_gpu_time": multi_gpu_time,
                "diff_single": diff_single,
                "diff_multi": diff_multi,
            }
        )

    return results


def plot_results(results):
    cpu_times = [r["cpu_time"] for r in results]
    single_gpu_times = [r["single_gpu_time"] for r in results]
    multi_gpu_times = [
        r["multi_gpu_time"] for r in results if r["multi_gpu_time"] is not None
    ]

    iterations = len(results)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, iterations + 1), cpu_times, label="CPU Time (ms)", marker="o")
    plt.plot(
        range(1, iterations + 1),
        single_gpu_times,
        label="Single GPU Time (ms)",
        marker="o",
    )
    if multi_gpu_times:
        plt.plot(
            range(1, len(multi_gpu_times) + 1),
            multi_gpu_times,
            label="Multi-GPU Time (ms)",
            marker="o",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Time (ms)")
    plt.title("Performance Comparison: CPU vs GPU(s)")
    plt.legend()
    plt.grid(True)
    plt.savefig("./img/performance_comparison.png")
    plt.close()
    print("\nPlot saved as 'performance_comparison.png'")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU performance.")
    parser.add_argument(
        "--iterations", type=int, default=1, help="Number of iterations to run the test"
    )
    args = parser.parse_args()

    results = test_speed(args.iterations)

    # Compute averages
    avg_cpu = sum(r["cpu_time"] for r in results) / args.iterations
    avg_gpu = sum(r["single_gpu_time"] for r in results) / args.iterations
    multi_gpu_available = any(r["multi_gpu_time"] is not None for r in results)
    avg_multi_gpu = (
        sum(r["multi_gpu_time"] for r in results if r["multi_gpu_time"] is not None)
        / sum(1 for r in results if r["multi_gpu_time"] is not None)
        if multi_gpu_available
        else None
    )

    print("\nAverage Results:")
    print(f"Average CPU Time: {avg_cpu:.2f} ms")
    print(f"Average Single GPU Time: {avg_gpu:.2f} ms")
    if avg_multi_gpu is not None:
        print(f"Average Multi-GPU Time: {avg_multi_gpu:.2f} ms")

    plot_results(results)


if __name__ == "__main__":
    main()
