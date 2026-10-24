Baseline (first lane): per-route
per-route: median 10.215 tok/s over 5 samples (receipt per-route)
grouped: median 10.741 tok/s over 5 samples (receipt grouped)
gnostral: median 23.645 tok/s over 5 samples (receipt gnostral)
llama.cpp: median 40.681 tok/s over 5 samples (receipt llama.cpp)

| Lane | Class | Placement | Decode tok/s (median) | Decode min-max | n | vs baseline range | Whole-card VRAM MiB (median) | Ambient VRAM MiB | Speed vs baseline | VRAM vs baseline | Pareto |
| --- | --- | --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | :---: |
| per-route | exact | 0-47:cuda:0;nm=cuda:0 | 10.215 | 9.756-10.497 | 5 | baseline | 3744.0 | UNKNOWN | 1.000x | 1.000x | DOMINATED |
| grouped | exact | 0-47:cuda:0;nm=cuda:0 | 10.741 | 10.204-10.885 | 5 | OVERLAP | 3729.0 | UNKNOWN | 1.051x | 0.996x | DOMINATED |
| gnostral | exact | 0-47:cuda:0;nm=cuda:0 | 23.645 | 20.372-24.263 | 5 | ABOVE | 3670.0 | UNKNOWN | 2.315x | 0.980x | FRONT |
| llama.cpp | self | llama.cpp offloaded=49/49 cpu_moe=on | 40.681 | 39.992-45.088 | 5 | ABOVE | 3945.0 | UNKNOWN | 3.982x | 1.054x | FRONT |

Self-class lanes run a different engine: they are gated on reproducing their own output, not on token identity with the baseline. Their agreement with the reference is in their run output, not ranked here.
