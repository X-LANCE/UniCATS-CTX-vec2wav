# #!/bin/bash

conda activate ctxv2w
export PATH=$PWD/utils:$PWD/ctx_vec2wav/bin:$PATH  # to add `ctx_vec2wav/bin` into PATH so that train.py and decode.py will directly be called
export PYTHONPATH=$PWD:$PYTHONPATH
