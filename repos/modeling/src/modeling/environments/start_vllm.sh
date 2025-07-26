#!/bin/bash

export NIX_LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export NIX_CFLAGS_COMPILE="-I/usr/local/cuda/include"
vllm serve ByteDance-Seed/UI-TARS-1.5-7B --enable-prefix-caching