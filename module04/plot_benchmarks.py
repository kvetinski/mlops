"""Plot Module 4 benchmark JSON reports from benchmark_api.py.

Usage:
    python module04/plot_benchmarks.py
    python module04/plot_benchmarks.py --reports-dir module04/reports --show

Dependencies: matplotlib
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_reports(directory: Path) -> list[dict]:
    files = sorted(directory.glob("c*.json"))
    if not files:
        raise SystemExit(f"No benchmark reports found in {directory} (expected c*.json)")

    reports = []
    for file in files:
        with file.open(encoding="utf-8") as stream:
            report = json.load(stream)
        required = (
            "concurrency",
            "requests",
            "failed_requests",
            "throughput_requests_per_second",
        )
        missing = [field for field in required if field not in report]
        if missing:
            raise ValueError(f"{file}: missing fields {missing}")
        reports.append(report)

    return sorted(reports, key=lambda report: report["concurrency"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Module 4 API benchmarks")
    parser.add_argument("--reports-dir", type=Path, default=Path("module04/reports"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show", action="store_true", help="Open plots interactively")
    args = parser.parse_args()

    reports = read_reports(args.reports_dir)
    output_dir = args.output_dir or args.reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    concurrencies = [report["concurrency"] for report in reports]

    # Plot successful-request end-to-end HTTP latencies.
    latency_reports = [r for r in reports if "successful_latency_ms" in r]
    if latency_reports:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = [r["concurrency"] for r in latency_reports]
        for metric in ("p50", "p95", "p99"):
            y = [r["successful_latency_ms"][metric] for r in latency_reports]
            ax.plot(x, y, marker="o", label=metric.upper())
        ax.set(title="HTTP latency vs concurrency", xlabel="Concurrent requests", ylabel="Latency (ms)")
        ax.set_xticks(concurrencies)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / "latency_vs_concurrency.png"
        fig.savefig(path, dpi=160)
        print(f"Saved {path}")
        if not args.show:
            plt.close(fig)
    else:
        print("No successful request latency samples; skipping latency chart")

    # Plot total completed requests per second (includes errors).
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        concurrencies,
        [r["throughput_requests_per_second"] for r in reports],
        marker="o",
    )
    ax.set(title="HTTP throughput vs concurrency", xlabel="Concurrent requests", ylabel="Requests/second")
    ax.set_xticks(concurrencies)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "throughput_vs_concurrency.png"
    fig.savefig(path, dpi=160)
    print(f"Saved {path}")
    if not args.show:
        plt.close(fig)

    # Always expose errors: successful-request latency could look fast despite failures.
    fig, ax = plt.subplots(figsize=(8, 5))
    failure_pct = [100.0 * r["failed_requests"] / r["requests"] for r in reports]
    ax.plot(concurrencies, failure_pct, marker="o")
    ax.set(title="HTTP failure rate vs concurrency", xlabel="Concurrent requests", ylabel="Failed requests (%)")
    ax.set_xticks(concurrencies)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "error_rate_vs_concurrency.png"
    fig.savefig(path, dpi=160)
    print(f"Saved {path}")
    if not args.show:
        plt.close(fig)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
