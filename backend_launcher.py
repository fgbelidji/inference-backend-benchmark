#!/usr/bin/env python3
"""
Launch one backend at a time, benchmark it, and stop it again.
Disabled backends can be listed via CLI:  --disable vllm,sglang
"""
import argparse, os, subprocess, time, yaml, json, shlex, socket, sys, signal, requests, threading

CONFIG = "bench-config.yaml"          # path to the file above
GPU_FLAG = "--gpus all"           # use every visible GPU
BENCHMARK_CMD = "inference-benchmarker"  # assume in PATH

def stream_container_logs(container_name):
    """Stream container logs in a separate thread"""
    def log_stream():
        try:
            process = subprocess.Popen(
                ["docker", "logs", "-f", container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
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
                s.connect((host, port)); return
            except OSError:
                time.sleep(1)
    raise TimeoutError(f"{host}:{port} never became ready")

def wait_server_ready(port, timeout=300):
    """Wait for the inference server to be fully ready with API endpoints"""
    deadline = time.time() + timeout
    base_url = f"http://localhost:{port}"
    
    # Different endpoints to check based on the server type
    endpoints_to_check = [         #
        "/health"
    ]
    
    while time.time() < deadline:
        for endpoint in endpoints_to_check:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                if response.status_code in [200, 404]:  # 404 is OK for some endpoints
                    print(f"✅ Server ready at {base_url}{endpoint}")
                    return
            except (requests.RequestException, requests.Timeout):
                continue
        print(f"⏳ Waiting for server at {base_url}...")
        time.sleep(3)
    
    raise TimeoutError(f"Server at {base_url} never became ready")

def run(cmd):
    print(cmd); subprocess.check_call(cmd, shell=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disable", default="")
    ap.add_argument("--show-logs", action="store_true", help="Show container logs")
    args = ap.parse_args()
    disabled = {x.strip() for x in args.disable.split(",") if x}

    cfg = yaml.safe_load(open(CONFIG))
    model = cfg["model"]
    backends = cfg["backends"]


    os.makedirs("results", exist_ok=True)

    for name, spec in backends.items():
        if name in disabled:
            print(f"Skipping {name}")
            continue

        image   = spec["image"]
        port    = spec["port"]
        arglist = spec["args"]
        # flatten the semi-colon notation into one list
        flat = [str(a) for part in arglist for a in (part if isinstance(part, list) else [part])]

        cmdline = " ".join(flat)
        


        # start container
        run(f"docker run --rm -d {GPU_FLAG} -p {port}:{port} --name {name} {image} {cmdline}")
        
        # Start streaming logs if requested
        log_thread = None
        if args.show_logs:
            log_thread = stream_container_logs(name)

        # wait until the TCP port answers
        print(f"⏳ Waiting for port {port} to be available...")
        wait_port("127.0.0.1", port)
        
        # wait until the server is fully ready
        print(f"⏳ Waiting for {name} server to be ready...")
        wait_server_ready(port)
        
        try:
            prompt_lengths = cfg["defaults"]["prompt_lengths"]
            bench_spec = cfg["defaults"]["bench"]
            # benchmark
            run(
                f"{BENCHMARK_CMD} "
                f"--url http://localhost:{port} "
                f"--tokenizer-name {model} "
                f"--benchmark-kind throughput "
                f"--duration {bench_spec['duration']} "
                f"--warmup {bench_spec['warmup']} "
                f"--max-vus {bench_spec['max_vus']} "
                f"--prompt-options num_tokens={prompt_lengths[0]},max_tokens={prompt_lengths[0]+4},min_tokens={prompt_lengths[0]-4},variance=0 "
                f"--decode-options num_tokens={bench_spec['decode_tokens']},max_tokens={bench_spec['decode_tokens']+4},min_tokens={bench_spec['decode_tokens']-4},variance=4 "
                f"--run-id {name} "
                f"--no-console"
                
            )
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
        finally:
            # stop container
            run(f"docker stop {name}")
            if log_thread:
                log_thread.join()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
