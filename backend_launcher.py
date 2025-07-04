#!/usr/bin/env python3
"""
Launch one backend at a time, benchmark it, and stop it again.
Disabled backends can be listed via CLI:  --disable vllm,sglang
"""
import argparse, os, subprocess, time, yaml, json, socket, sys, requests, threading, uuid
from datetime import datetime

CONFIG = "bench-config.yaml" 
GPU_FLAG = "--gpus all"  
BENCHMARK_CMD = "vllm bench serve"


def stream_container_logs(container_name):
    """Stream container logs in a separate thread"""

    def log_stream():
        try:
            process = subprocess.Popen(
                ["docker", "logs", "-f", container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            for line in process.stdout:
                print(f"[{container_name}] {line.rstrip()}")
        except Exception as e:
            print(f"Error streaming logs for {container_name}: {e}")

    thread = threading.Thread(target=log_stream, daemon=True)
    thread.start()
    return thread


def wait_port(host, port, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            try:
                s.connect((host, port))
                return
            except OSError:
                time.sleep(1)
    raise TimeoutError(f"{host}:{port} never became ready")


def wait_server_ready(port, timeout=300):
    """Wait for the inference server to be fully ready with API endpoints"""
    deadline = time.time() + timeout
    base_url = f"http://localhost:{port}"

    # Different endpoints to check based on the server type
    endpoints_to_check = ["/health"]  #

    while time.time() < deadline:
        for endpoint in endpoints_to_check:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                if response.status_code in [200, 404]:  # 404 is OK for some endpoints
                    print(f"Server ready at {base_url}{endpoint}")
                    return
            except (requests.RequestException, requests.Timeout):
                continue
        print(f"Waiting for server at {base_url}...")
        time.sleep(3)

    raise TimeoutError(f"Server at {base_url} never became ready")


def run(cmd):
    print(cmd)
    subprocess.check_call(cmd, shell=True)


def format_metadata_from_config(config, prefix=""):
    """Convert configuration dictionary to metadata format"""
    metadata_items = []

    def flatten_value(value):
        """Convert value to string format suitable for metadata"""
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        elif isinstance(value, dict):
            return json.dumps(value)
        else:
            return str(value)

    for key, value in config.items():
        # Convert key to metadata format (replace dashes with underscores)
        metadata_key = key.replace("-", "_")
        if prefix:
            metadata_key = f"{prefix}_{metadata_key}"

        # Handle different value types
        if isinstance(value, (str, int, float, bool)):
            metadata_items.append(f"{metadata_key}={value}")
        elif isinstance(value, list):
            # For lists, join with commas and escape any spaces
            list_str = ",".join(str(item) for item in value)
            metadata_items.append(f"{metadata_key}={list_str}")
        elif isinstance(value, dict):
            # For nested dictionaries, convert to JSON
            metadata_items.append(f"{metadata_key}={json.dumps(value)}")
        else:
            metadata_items.append(f"{metadata_key}={str(value)}")

    return metadata_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disable", default="")
    ap.add_argument("--show-logs", action="store_true", help="Show container logs")
    args = ap.parse_args()
    disabled = {x.strip() for x in args.disable.split(",") if x}

    cfg = yaml.safe_load(open(CONFIG))
    model = cfg["model"]
    backends = cfg["backends"]
    benchmark = cfg["benchmark"]

    # Generate unique run identifier
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_uuid = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    run_id = f"{run_timestamp}-{run_uuid}"

    print(f"Starting benchmark run with ID: {run_id}")
    print(f"Model: {model}")
    print(f"Backends: {', '.join(backends.keys())}")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)

    for name, spec in backends.items():
        if name in disabled:
            print(f"Skipping {name}")
            continue

        print(f"\nStarting benchmark for backend: {name}")
        print(f"Image: {spec['image']}")
        print(f"Port: {spec['port']}")

        image = spec["image"]
        port = spec["port"]
        arglist = spec["args"]
        # flatten the semi-colon notation into one list
        flat = [
            str(a)
            for part in arglist
            for a in (part if isinstance(part, list) else [part])
        ]

        cmdline = " ".join(flat)

        # print(f"docker run --rm -d {GPU_FLAG} -p {port}:{port} --shm-size 32G --name {name} {image} {cmdline}")
        # start container
        run(
            f"docker run --rm -d {GPU_FLAG} -p {port}:{port} --shm-size 32G --name {name} {image} {cmdline}"
        )

        # Start streaming logs if requested
        log_thread = None
        if args.show_logs:
            log_thread = stream_container_logs(name)

        # wait until the TCP port answers
        print(f"Waiting for port {port} to be available...")
        wait_port("127.0.0.1", port)

        # wait until the server is fully ready
        print(f"Waiting for {name} server to be ready...")
        wait_server_ready(port)

        try:
            benchmark_args = [f"--{k} {v}" for k, v in benchmark.items()]
            benchmark_args_line = " ".join(benchmark_args)

            # Create result directory with run ID
            result_dir = f"./results/{model.replace('/', '-')}/{name}/{run_id}/"
            os.makedirs(result_dir, exist_ok=True)

            # Prepare metadata from benchmark configurations
            benchmark_metadata = format_metadata_from_config(benchmark, "benchmark")

            # Combine all metadata
            all_metadata = [
                f"run_id={run_id}",
                f"backend={name}",
                f"timestamp={run_timestamp}",
                f"model={model}",
            ] + benchmark_metadata

            # Create separate --metadata flags for each item
            metadata_flags = " ".join([f"--metadata {item}" for item in all_metadata])

            # benchmark
            run(
                f"{BENCHMARK_CMD} "
                f"--endpoint-type openai "
                f"--host localhost "
                f"--port {port} "
                f"--model {model} "
                f"--result-dir {result_dir} "
                f"--save-result "
                f"{metadata_flags} "
                f"{benchmark_args_line}"
            )
            print(f"Benchmark completed for {name}")
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
        finally:
            # stop container
            print(f" Stopping {name} container")
            run(f"docker stop {name}")
            if log_thread:
                log_thread.join()

    print(f"\n Benchmark run {run_id} completed!")
    print(f"Results saved in: ./results/{model.replace('/', '-')}/")
    print(
        f"Use: python summarize_benchmarks.py {model.replace('/', '-')} to view summary"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
