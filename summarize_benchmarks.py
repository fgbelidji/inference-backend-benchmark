"""
Summarize benchmark results for a given model.
Usage: python summarize_benchmarks.py [model_name]
"""

import argparse
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime


class BenchmarkSummarizer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

    def get_available_models(self) -> List[str]:
        """Get list of available models from results directory."""
        models = []
        if self.results_dir.exists():
            for model_dir in self.results_dir.iterdir():
                if model_dir.is_dir():
                    models.append(model_dir.name)
        return sorted(models)

    def load_benchmark_data(
        self, model_name: str, run_id: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """Load all benchmark data for a specific model."""
        model_dir = self.results_dir / model_name
        if not model_dir.exists():
            raise ValueError(f"Model '{model_name}' not found in results directory")

        data = {}
        for backend_dir in model_dir.iterdir():
            if backend_dir.is_dir():
                backend_name = backend_dir.name
                data[backend_name] = []

                # Load all JSON files for this backend (including subdirectories with run IDs)
                for json_file in backend_dir.glob("**/*.json"):
                    try:
                        with open(json_file, "r") as f:
                            result = json.load(f)
                            result["file_path"] = str(json_file)
                            # Extract run_id from path if available
                            path_parts = json_file.parts
                            if (
                                len(path_parts) > 3
                            ):  # results/model/backend/run_id/file.json
                                result["run_id"] = path_parts[-2]

                            # Filter by run_id if specified
                            if run_id is None or result.get("run_id") == run_id:
                                data[backend_name].append(result)
                    except Exception as e:
                        print(f"Warning: Could not load {json_file}: {e}")

        return data

    def get_latest_results(self, backend_data: List[Dict]) -> Optional[Dict]:
        """Get the most recent benchmark result for a backend."""
        if not backend_data:
            return None

        # Sort by date and return the most recent
        return max(backend_data, key=lambda x: x.get("date", ""))

    def format_metrics_table(self, data: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Create a formatted table comparing metrics across backends."""
        rows = []

        for backend_name, backend_data in data.items():
            latest = self.get_latest_results(backend_data)
            if latest:
                row = {
                    "Backend": backend_name,
                    "Run ID": latest.get("run_id", "N/A"),
                    "Date": latest.get("date", "N/A"),
                    "Request Rate (req/s)": f"{latest.get('request_rate', 0):.1f}",
                    "Request Throughput (req/s)": f"{latest.get('request_throughput', 0):.2f}",
                    "Output Throughput (tokens/s)": f"{latest.get('output_throughput', 0):.2f}",
                    "Total Token Throughput (tokens/s)": f"{latest.get('total_token_throughput', 0):.2f}",
                    "Mean TTFT (ms)": f"{latest.get('mean_ttft_ms', 0):.2f}",
                    "P99 TTFT (ms)": f"{latest.get('p99_ttft_ms', 0):.2f}",
                    "Mean TPOT (ms)": f"{latest.get('mean_tpot_ms', 0):.2f}",
                    "P99 TPOT (ms)": f"{latest.get('p99_tpot_ms', 0):.2f}",
                    "Duration (s)": f"{latest.get('duration', 0):.2f}",
                    "Completed Requests": latest.get("completed", 0),
                    "Total Input Tokens": latest.get("total_input_tokens", 0),
                    "Total Output Tokens": latest.get("total_output_tokens", 0),
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def get_performance_ranking(
        self, data: Dict[str, List[Dict]]
    ) -> Dict[str, List[tuple]]:
        """Rank backends by different performance metrics."""
        metrics = {}

        for backend_name, backend_data in data.items():
            latest = self.get_latest_results(backend_data)
            if latest:
                metrics[backend_name] = {
                    "output_throughput": latest.get("output_throughput", 0),
                    "total_token_throughput": latest.get("total_token_throughput", 0),
                    "mean_ttft_ms": latest.get("mean_ttft_ms", float("inf")),
                    "mean_tpot_ms": latest.get("mean_tpot_ms", float("inf")),
                    "p99_ttft_ms": latest.get("p99_ttft_ms", float("inf")),
                    "p99_tpot_ms": latest.get("p99_tpot_ms", float("inf")),
                }

        rankings = {}

        # Throughput rankings (higher is better)
        for metric in ["output_throughput", "total_token_throughput"]:
            rankings[f"Best {metric.replace('_', ' ').title()}"] = sorted(
                metrics.items(), key=lambda x: x[1][metric], reverse=True
            )

        # Latency rankings (lower is better)
        for metric in ["mean_ttft_ms", "mean_tpot_ms", "p99_ttft_ms", "p99_tpot_ms"]:
            rankings[
                f"Best {metric.replace('_', ' ').replace('ms', '(ms)').title()}"
            ] = sorted(metrics.items(), key=lambda x: x[1][metric])

        return rankings

    def print_summary(
        self,
        model_name: str,
        data: Dict[str, List[Dict]],
        output_file: Optional[str] = None,
    ):
        """Print a comprehensive summary of benchmark results in markdown format."""
        import sys
        from io import StringIO

        # Capture output if saving to file
        if output_file:
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

        print(f"# Benchmark Summary for {model_name}")
        print()

        # Basic info
        total_backends = len(data)
        total_runs = sum(len(runs) for runs in data.values())

        print(f"**Total Backends Tested:** {total_backends}")
        print(f"**Total Benchmark Runs:** {total_runs}")
        print(f"**Available Backends:** {', '.join(data.keys())}")
        print()

        # Performance table
        print(f"## Performance Comparison (Latest Results)")
        print()

        df = self.format_metrics_table(data)
        if not df.empty:
            try:
                print(df.to_markdown(index=False))
            except ImportError:
                print("Note: Install 'tabulate' for better table formatting")
                print()
                print("| " + " | ".join(df.columns) + " |")
                print("| " + " | ".join(["---"] * len(df.columns)) + " |")
                for _, row in df.iterrows():
                    print("| " + " | ".join(str(val) for val in row) + " |")
        else:
            print("No benchmark data available")
            return

        # Performance rankings
        print(f"\n## Performance Rankings")
        print()

        rankings = self.get_performance_ranking(data)

        # Create a proper mapping for the metric keys
        key_mapping = {
            "Best Output Throughput": "output_throughput",
            "Best Total Token Throughput": "total_token_throughput",
            "Best Mean Ttft (Ms)": "mean_ttft_ms",
            "Best Mean Tpot (Ms)": "mean_tpot_ms",
            "Best P99 Ttft (Ms)": "p99_ttft_ms",
            "Best P99 Tpot (Ms)": "p99_tpot_ms",
        }

        for rank_name, ranked_backends in rankings.items():
            print(f"\n### {rank_name}")
            for i, (backend, metrics) in enumerate(ranked_backends, 1):
                metric_key = key_mapping.get(
                    rank_name,
                    rank_name.lower()
                    .replace("best ", "")
                    .replace(" ", "_")
                    .replace("(ms)", "_ms"),
                )
                metric_value = metrics.get(metric_key, 0)
                if "ms" in metric_key:
                    print(f"{i}. **{backend}**: {metric_value:.2f} ms")
                else:
                    print(f"{i}. **{backend}**: {metric_value:.2f}")

        # Historical data summary
        print(f"\n## Historical Data Summary")
        print()

        for backend_name, backend_data in data.items():
            if len(backend_data) > 1:
                dates = [run.get("date", "") for run in backend_data]
                run_ids = [run.get("run_id", "N/A") for run in backend_data]
                print(
                    f"**{backend_name}**: {len(backend_data)} runs from {min(dates)} to {max(dates)}"
                )
                print(f"  - Run IDs: {', '.join(run_ids)}")
            else:
                run_id = backend_data[0].get("run_id", "N/A")
                print(f"**{backend_name}**: {len(backend_data)} run (ID: {run_id})")

        # Key insights
        print(f"\n## Key Insights")
        print()

        if not df.empty:
            # Find best performers
            best_throughput = df.loc[
                df["Output Throughput (tokens/s)"].astype(float).idxmax(), "Backend"
            ]
            best_ttft = df.loc[df["Mean TTFT (ms)"].astype(float).idxmin(), "Backend"]
            best_tpot = df.loc[df["Mean TPOT (ms)"].astype(float).idxmin(), "Backend"]

            print(f"- **Highest throughput**: {best_throughput}")
            print(f"- **Lowest time-to-first-token**: {best_ttft}")
            print(f"- **Lowest time-per-output-token**: {best_tpot}")
            print()

            # Performance spread
            throughput_values = df["Output Throughput (tokens/s)"].astype(float)
            ttft_values = df["Mean TTFT (ms)"].astype(float)

            print(
                f"- **Throughput range**: {throughput_values.min():.1f} - {throughput_values.max():.1f} tokens/s"
            )
            print(
                f"- **TTFT range**: {ttft_values.min():.1f} - {ttft_values.max():.1f} ms"
            )

        # Save to file if requested
        if output_file:
            content = captured_output.getvalue()
            sys.stdout = old_stdout
            with open(output_file, "w") as f:
                f.write(content)
            print(f"Markdown report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize benchmark results for a given model"
    )
    parser.add_argument("model", nargs="?", help='Model name (e.g., "Qwen-Qwen3-0.6B")')
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--results-dir", default="results", help="Path to results directory"
    )
    parser.add_argument("--run-id", help="Filter results by specific run ID")
    parser.add_argument("--output", "-o", help="Save markdown report to file")

    args = parser.parse_args()

    summarizer = BenchmarkSummarizer(args.results_dir)

    if args.list_models:
        models = summarizer.get_available_models()
        print("Available models:")
        for model in models:
            print(f"  - {model}")
        return

    if not args.model:
        models = summarizer.get_available_models()
        if not models:
            print("No benchmark results found in results directory")
            return

        print("Available models:")
        for model in models:
            print(f"  - {model}")
        print(f"\nUsage: python {__file__} <model_name>")
        return

    try:
        data = summarizer.load_benchmark_data(args.model, args.run_id)
        if args.run_id:
            print(f"Filtering results for run ID: {args.run_id}")
        summarizer.print_summary(args.model, data, args.output)
    except ValueError as e:
        print(f"Error: {e}")
        models = summarizer.get_available_models()
        if models:
            print(f"Available models: {', '.join(models)}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
