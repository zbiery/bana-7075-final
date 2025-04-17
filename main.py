import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import uvicorn


def run_mlflow():
    """Start the MLflow server based on the OS."""
    system_os = platform.system()

    try:
        if system_os == "Windows":
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", "./mlflow_server.ps1"], check=True)
        elif system_os in ["Linux", "Darwin"]:
            subprocess.run(["bash", "./mlflow_server.sh"], check=True)
        else:
            raise EnvironmentError(f"Unsupported OS: {system_os}")
    except subprocess.CalledProcessError as e:
        print(f"[MLflow] Failed to start MLflow server: {e}")


def run_fastapi():
    """Start the FastAPI app."""
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=1000)


def run_streamlit():
    """Start the Streamlit app."""
    os.system("streamlit run streamlit_app.py")


def shutdown_handler(signum, frame):
    """Gracefully terminate child processes."""
    print("\nShutting down MLflow, FastAPI, and Streamlit...")

    for p in processes:
        p.terminate()
        p.join()

    sys.exit(0)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    # Define all processes
    processes = [
        multiprocessing.Process(target=run_mlflow, daemon=True),
        multiprocessing.Process(target=run_fastapi, daemon=True),
        multiprocessing.Process(target=run_streamlit, daemon=True)
    ]

    # Start all processes
    for p in processes:
        p.start()

    # Capture termination signals
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        shutdown_handler(None, None)
