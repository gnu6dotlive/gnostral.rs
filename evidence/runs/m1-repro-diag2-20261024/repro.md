| config | prompt | lp idx | tok idx | gap prev r1 | gap prev r2 | gap@div r1 | gap@div r2 | max lp diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | q04-units | - | - | - | - | - | - | 0.00e+00 |
| A | q07-sql | - | - | - | - | - | - | 0.00e+00 |
| A | q09-bug | - | - | - | - | - | - | 0.00e+00 |
| A | q10-fr-explain | - | - | - | - | - | - | 0.00e+00 |
| A | q11-fr-translate | - | - | - | - | - | - | 0.00e+00 |
| A | q19-summary-tech | - | - | - | - | - | - | 0.00e+00 |
| A | q27-science | - | - | - | - | - | - | 0.00e+00 |
| A | q29-code-explain | - | - | - | - | - | - | 0.00e+00 |
| A | q32-refusal-boundary | - | - | - | - | - | - | 0.00e+00 |
| B | q04-units | - | - | - | - | - | - | 0.00e+00 |
| B | q07-sql | - | - | - | - | - | - | 0.00e+00 |
| B | q09-bug | - | - | - | - | - | - | 0.00e+00 |
| B | q10-fr-explain | - | - | - | - | - | - | 0.00e+00 |
| B | q11-fr-translate | - | - | - | - | - | - | 0.00e+00 |
| B | q19-summary-tech | - | - | - | - | - | - | 0.00e+00 |
| B | q27-science | - | - | - | - | - | - | 0.00e+00 |
| B | q29-code-explain | - | - | - | - | - | - | 0.00e+00 |
| B | q32-refusal-boundary | - | - | - | - | - | - | 0.00e+00 |
| C1 | q04-units | 1 | 27 | 1.71e+01 | 1.76e+01 | 0.00e+00 | 2.50e-01 | 3.05e-01 |
| C1 | q07-sql | 0 | 16 | - | - | 2.50e-01 | 1.00e+00 | 6.61e-02 |
| C1 | q09-bug | 0 | 24 | - | - | 7.50e-01 | 2.50e-01 | 2.06e-01 |
| C1 | q10-fr-explain | 0 | 23 | - | - | 1.00e+00 | 0.00e+00 | 7.36e-02 |
| C1 | q11-fr-translate | 0 | 10 | - | - | 7.50e-01 | 0.00e+00 | 7.49e-02 |
| C1 | q19-summary-tech | 0 | 8 | - | - | 2.50e-01 | 0.00e+00 | 9.25e-04 |
| C1 | q27-science | 0 | 29 | - | - | 0.00e+00 | 2.50e-01 | 1.19e-01 |
| C1 | q29-code-explain | 0 | 21 | - | - | 0.00e+00 | 5.00e-01 | 1.89e-01 |
| C1 | q32-refusal-boundary | 0 | 23 | - | - | 2.50e-01 | 0.00e+00 | 5.01e-02 |
| C2 | q04-units | - | - | - | - | - | - | 0.00e+00 |
| C2 | q07-sql | - | - | - | - | - | - | 0.00e+00 |
| C2 | q09-bug | - | - | - | - | - | - | 0.00e+00 |
| C2 | q10-fr-explain | - | - | - | - | - | - | 0.00e+00 |
| C2 | q11-fr-translate | - | - | - | - | - | - | 0.00e+00 |
| C2 | q19-summary-tech | - | - | - | - | - | - | 0.00e+00 |
| C2 | q27-science | - | - | - | - | - | - | 0.00e+00 |
| C2 | q29-code-explain | - | - | - | - | - | - | 0.00e+00 |
| C2 | q32-refusal-boundary | - | - | - | - | - | - | 0.00e+00 |
| D | q04-units | 1 | 27 | 1.76e+01 | 1.71e+01 | 2.50e-01 | 0.00e+00 | 3.05e-01 |
| D | q07-sql | 0 | 16 | - | - | 1.00e+00 | 2.50e-01 | 6.61e-02 |
| D | q09-bug | 0 | 24 | - | - | 2.50e-01 | 7.50e-01 | 2.06e-01 |
| D | q10-fr-explain | 0 | 23 | - | - | 0.00e+00 | 1.00e+00 | 7.36e-02 |
| D | q11-fr-translate | 0 | 10 | - | - | 0.00e+00 | 7.50e-01 | 7.49e-02 |
| D | q19-summary-tech | 0 | 8 | - | - | 0.00e+00 | 2.50e-01 | 9.25e-04 |
| D | q27-science | 0 | 29 | - | - | 2.50e-01 | 0.00e+00 | 1.19e-01 |
| D | q29-code-explain | 0 | 21 | - | - | 5.00e-01 | 0.00e+00 | 1.89e-01 |
| D | q32-refusal-boundary | 0 | 23 | - | - | 0.00e+00 | 2.50e-01 | 5.01e-02 |

A: tokens differ 0/9, logprobs differ 0/9, device map run1 ['Layers 0-15: cuda[0]', 'Layers 16-47: cpu'] run2 ['Layers 0-15: cuda[0]', 'Layers 16-47: cpu']

B: tokens differ 0/9, logprobs differ 0/9, device map run1 ['Layers 0-47: cpu'] run2 ['Layers 0-47: cpu']

C1: tokens differ 9/9, logprobs differ 9/9, device map run1 ['Layers 0-15: cuda[0]', 'Layers 16-47: cpu'] run2 ['Layers 0-14: cuda[0]', 'Layers 15-47: cpu']

C2: tokens differ 0/9, logprobs differ 0/9, device map run1 ['Layers 0-14: cuda[0]', 'Layers 15-47: cpu'] run2 ['Layers 0-14: cuda[0]', 'Layers 15-47: cpu']

D: tokens differ 9/9, logprobs differ 9/9, device map run1 ['Layers 0-14: cuda[0]', 'Layers 15-47: cpu'] run2 ['Layers 0-15: cuda[0]', 'Layers 16-47: cpu']