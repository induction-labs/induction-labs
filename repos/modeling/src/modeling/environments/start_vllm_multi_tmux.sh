#!/usr/bin/env bash
# launch_vllm_tmux.sh  -- all workers started via tmux send‑keys
# -------------------------------------------------------------
# 0. environment needed by vLLM / CUDA
export NIX_LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export NIX_CFLAGS_COMPILE="-I/usr/local/cuda/include"

# 1. cluster parameters
SESSION="vllm_cluster"
MODEL="ByteDance-Seed/UI-TARS-1.5-7B"
BASE_PORT=8000          # 8000‑8007
NUM_GPUS=8

# 2. create a new detached tmux session with the first window (empty command)
tmux new-session -d -s "$SESSION" -n "gpu0"

# 3. keep panes open after commands finish
tmux set-option -t "$SESSION" remain-on-exit on

# 4. enqueue the first worker command with send‑keys
tmux send-keys -t "$SESSION:gpu0" \
  "CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL \
   --port $BASE_PORT \
   --enable-prefix-caching \
   --tensor-parallel-size 1" C-m

# 5. create the remaining worker windows & send their commands
for GPU in $(seq 1 $((NUM_GPUS-1))); do
  PORT=$((BASE_PORT + GPU))
  WIN="gpu$GPU"
  tmux new-window -t "$SESSION" -n "$WIN"
  tmux send-keys -t "$SESSION:$WIN" \
    "CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL \
     --port $PORT \
     --enable-prefix-caching \
     --tensor-parallel-size 1" C-m
done

# 6. optional FastAPI proxy window (also via send‑keys)
tmux new-window -t "$SESSION" -n "proxy"
tmux send-keys -t "$SESSION:proxy" \
  "uv run fastapi run src/modeling/environments/load_balance_server.py --port 8080" C-m

# 7. attach so you can watch logs (Ctrl‑b d to detach)
tmux attach-session -t "$SESSION"
