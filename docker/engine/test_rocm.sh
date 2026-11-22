#!/bin/bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
python3 << 'PYEOF'
import onnxruntime as ort
import onnxruntime_ep_migraphx as m
print('EP names:', m.get_ep_names())
print('Library paths:', m.get_library_paths())
[ort.register_execution_provider_library(n,p) for n,p in zip(m.get_ep_names(), m.get_library_paths())]
print('Providers:', ort.get_available_providers())
PYEOF
