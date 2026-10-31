# Source this AFTER activating the venv, in every terminal that runs the worker or any GPU code:
#   source .venv/bin/activate && source gpu_env.sh
SP=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export LD_LIBRARY_PATH="$SP/nvidia/cublas/lib:$SP/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
