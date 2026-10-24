Baseline (first lane): per-route
per-route: median 16.304 tok/s over 5 samples (receipt per-route)
grouped: median 18.724 tok/s over 5 samples (receipt grouped)
gnostral: median 28.869 tok/s over 5 samples (receipt gnostral)
llama.cpp: median 44.725 tok/s over 5 samples (receipt llama.cpp)

| Lane | Class | Placement | Decode tok/s (median) | Decode min-max | n | vs baseline range | Whole-card VRAM MiB (median) | Ambient VRAM MiB | Speed vs baseline | VRAM vs baseline | Pareto |
| --- | --- | --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | :---: |
| per-route | exact | 0-47:cuda:0;nm=cuda:0 | 16.304 | 13.381-16.701 | 5 | baseline | 5943.0 | UNKNOWN | 1.000x | 1.000x | DOMINATED |
| grouped | exact | 0-47:cuda:0;nm=cuda:0 | 18.724 | 17.610-18.880 | 5 | ABOVE | 5921.0 | UNKNOWN | 1.148x | 0.996x | DOMINATED |
| gnostral | exact | 0-47:cuda:0;nm=cuda:0 | 28.869 | 28.595-30.270 | 5 | ABOVE | 5756.0 | UNKNOWN | 1.771x | 0.969x | DOMINATED |
| llama.cpp | self | llama.cpp offloaded=49/49 cpu_moe=on | 44.725 | 42.730-47.043 | 5 | ABOVE | 3943.0 | UNKNOWN | 2.743x | 0.663x | FRONT |

Self-class lanes run a different engine: they are gated on reproducing their own output, not on token identity with the baseline. Their agreement with the reference is in their run output, not ranked here.
