# GP-BO reward landscape

GP-BO was not rerun and was not modified. Statistics come from `history` already saved in each family JSON.

## Model mode

- Digital twin `training` flag at load: `False`.
- `load_twin` calls `model.eval()` and freezes weights. `BatteryDigitalTwin.predict` calls `eval()` again, and chunk inference is under `torch.no_grad()`.
- Dropout modules: 4. All in eval after load: `True`.
- The object passed into `optimize_family_gp_bo` as `model` is `HybridDegradationModel`, a closed-form NumPy scorer. It is not an `nn.Module` and has no `eval()` / `train()` switch.
- Repeated-evaluation device: `cuda:0`.

## Repeated evaluation of one fixed profile

- Cell `RW9`, family `cccv`, 10 identical calls.
- Reward std: `0.000e+00`. Reward max−min: `0.000e+00`.
- Loss std: `5.551e-17`.
- Deterministic at 1e-12 absolute tolerance: `True`.

## Loss is not exactly minus reward

`score_session` sets `loss = -reward + 0.001 * duration_min + penalty`.
A lower loss usually tracks a higher reward, but the duration term and the infeasibility penalty can pick a different winner.

## 02_armA — original reward

| Cell | Family | N | Feasible | Min | P25 | Median | Mean | P75 | Max | Std | Within 1% of best | Loss winner = reward winner |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RW9 | CCCV (feasible) | 78 | 78 | 0.288908 | 0.298248 | 0.302999 | 0.302046 | 0.306949 | 0.311643 | 0.006514 | 15/78 | True |
| RW9 | Two-step (feasible) | 66 | 66 | 0.291009 | 0.299860 | 0.305461 | 0.303367 | 0.308550 | 0.310723 | 0.006268 | 18/66 | False |
| RW9 | Three-step (feasible) | 56 | 56 | 0.293008 | 0.299787 | 0.303163 | 0.302825 | 0.306847 | 0.310348 | 0.004706 | 12/56 | False |
| RW9 | Pulsed (feasible) | 51 | 51 | 0.233718 | 0.260295 | 0.267951 | 0.268712 | 0.279877 | 0.298029 | 0.015663 | 1/51 | True |
| RW10 | CCCV (feasible) | 69 | 69 | 0.276445 | 0.297780 | 0.302258 | 0.301524 | 0.305865 | 0.312697 | 0.007546 | 12/69 | False |
| RW10 | Two-step (feasible) | 65 | 65 | 0.287970 | 0.296703 | 0.303624 | 0.302667 | 0.308607 | 0.312799 | 0.007530 | 13/65 | False |
| RW10 | Three-step (feasible) | 68 | 68 | 0.291362 | 0.301131 | 0.305633 | 0.305386 | 0.311161 | 0.312563 | 0.005725 | 24/68 | False |
| RW10 | Pulsed (feasible) | 55 | 55 | 0.238853 | 0.257704 | 0.270013 | 0.271486 | 0.287572 | 0.306219 | 0.018360 | 2/55 | True |
| RW11 | CCCV (feasible) | 78 | 78 | 0.299916 | 0.330270 | 0.331450 | 0.328835 | 0.331873 | 0.333069 | 0.007870 | 64/78 | False |
| RW11 | Two-step (feasible) | 75 | 75 | 0.301634 | 0.326523 | 0.330226 | 0.326937 | 0.331613 | 0.332142 | 0.007662 | 48/75 | False |
| RW11 | Three-step (feasible) | 69 | 69 | 0.308907 | 0.322197 | 0.329506 | 0.326614 | 0.331198 | 0.331975 | 0.005753 | 40/69 | False |
| RW11 | Pulsed (feasible) | 55 | 55 | 0.247750 | 0.286557 | 0.291359 | 0.294061 | 0.308184 | 0.322353 | 0.016542 | 2/55 | True |
| RW12 | CCCV (feasible) | 64 | 64 | 0.311116 | 0.312606 | 0.313785 | 0.321814 | 0.316933 | 0.364786 | 0.017603 | 8/64 | False |
| RW12 | Two-step (feasible) | 57 | 57 | 0.311394 | 0.312362 | 0.313104 | 0.317158 | 0.316607 | 0.351451 | 0.009284 | 2/57 | False |
| RW12 | Three-step (feasible) | 48 | 48 | 0.307088 | 0.311486 | 0.312350 | 0.313511 | 0.315004 | 0.330991 | 0.003937 | 1/48 | False |
| RW12 | Pulsed (feasible) | 56 | 56 | 0.237708 | 0.277047 | 0.289904 | 0.290487 | 0.302042 | 0.348470 | 0.026573 | 1/56 | False |

Top 6 feasible candidates per cell, ranked by loss (the rule GP-BO uses to pick a winner):

### RW9

| Rank | Family | Eval | Loss | Reward | Δreward from best feasible | Within 1% |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | two_step | 23 | -0.282861 | 0.310545 | 0.001098 | True |
| 2 | cccv | 34 | -0.282793 | 0.311643 | 0.000000 | True |
| 3 | cccv | 63 | -0.282719 | 0.310452 | 0.001191 | True |
| 4 | cccv | 19 | -0.282547 | 0.310897 | 0.000746 | True |
| 5 | two_step | 61 | -0.282489 | 0.310356 | 0.001288 | True |
| 6 | two_step | 66 | -0.282350 | 0.310450 | 0.001194 | True |

Feasible candidates within 1% of this cell's best feasible reward: **37 / 251** (best reward 0.311643).

### RW10

| Rank | Family | Eval | Loss | Reward | Δreward from best feasible | Within 1% |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | two_step | 40 | -0.288291 | 0.312424 | 0.000375 | True |
| 2 | two_step | 67 | -0.288168 | 0.312518 | 0.000281 | True |
| 3 | three_step | 34 | -0.287905 | 0.312255 | 0.000545 | True |
| 4 | cccv | 1 | -0.287893 | 0.312610 | 0.000189 | True |
| 5 | two_step | 2 | -0.287893 | 0.312610 | 0.000189 | True |
| 6 | three_step | 73 | -0.287722 | 0.312338 | 0.000461 | True |

Feasible candidates within 1% of this cell's best feasible reward: **48 / 257** (best reward 0.312799).

### RW11

| Rank | Family | Eval | Loss | Reward | Δreward from best feasible | Within 1% |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | cccv | 30 | -0.317202 | 0.332202 | 0.000867 | True |
| 2 | cccv | 62 | -0.316986 | 0.333069 | 0.000000 | True |
| 3 | cccv | 74 | -0.316902 | 0.331952 | 0.001117 | True |
| 4 | cccv | 22 | -0.316854 | 0.331837 | 0.001232 | True |
| 5 | cccv | 17 | -0.316633 | 0.332866 | 0.000203 | True |
| 6 | cccv | 32 | -0.316522 | 0.331555 | 0.001514 | True |

Feasible candidates within 1% of this cell's best feasible reward: **137 / 277** (best reward 0.333069).

### RW12

| Rank | Family | Eval | Loss | Reward | Δreward from best feasible | Within 1% |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | two_step | 68 | -0.290286 | 0.311769 | 0.053017 | False |
| 2 | two_step | 42 | -0.290277 | 0.311561 | 0.053225 | False |
| 3 | cccv | 29 | -0.290243 | 0.311876 | 0.052910 | False |
| 4 | two_step | 72 | -0.290195 | 0.311628 | 0.053157 | False |
| 5 | two_step | 43 | -0.289968 | 0.311768 | 0.053018 | False |
| 6 | three_step | 28 | -0.289960 | 0.311493 | 0.053292 | False |

Feasible candidates within 1% of this cell's best feasible reward: **8 / 225** (best reward 0.364786).

## 03_armA_rescaled — rescaled reward (normalize_three_terms)

| Cell | Family | N | Feasible | Min | P25 | Median | Mean | P75 | Max | Std | Within 1% of best | Loss winner = reward winner |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RW9 | CCCV (feasible) | 72 | 72 | -1.407252 | -0.982384 | -0.862567 | -0.876282 | -0.763046 | -0.652315 | 0.162559 | 3/72 | False |
| RW9 | Two-step (feasible) | 66 | 66 | -1.324611 | -0.921858 | -0.731075 | -0.819681 | -0.664824 | -0.635534 | 0.207473 | 5/66 | False |
| RW9 | Three-step (feasible) | 49 | 49 | -1.170948 | -0.964430 | -0.861595 | -0.869747 | -0.755447 | -0.664966 | 0.124577 | 1/49 | False |
| RW9 | Pulsed (feasible) | 57 | 57 | -2.264175 | -1.909851 | -1.267737 | -1.456826 | -1.073366 | -0.865417 | 0.428084 | 1/57 | False |
| RW10 | CCCV (feasible) | 71 | 71 | -1.398613 | -0.937397 | -0.825248 | -0.842329 | -0.726025 | -0.588306 | 0.181231 | 1/71 | True |
| RW10 | Two-step (feasible) | 64 | 64 | -1.329270 | -0.990205 | -0.729661 | -0.826011 | -0.649282 | -0.596684 | 0.209352 | 2/64 | True |
| RW10 | Three-step (feasible) | 68 | 68 | -1.203118 | -0.815901 | -0.680344 | -0.735262 | -0.619127 | -0.593578 | 0.144545 | 7/68 | True |
| RW10 | Pulsed (feasible) | 61 | 61 | -2.451372 | -1.946578 | -1.798004 | -1.671229 | -1.399480 | -0.821064 | 0.396199 | 1/61 | True |
| RW11 | CCCV (feasible) | 78 | 78 | -1.418153 | -0.464866 | -0.456080 | -0.502364 | -0.453994 | -0.449510 | 0.165065 | 20/78 | True |
| RW11 | Two-step (feasible) | 74 | 74 | -1.338582 | -0.502113 | -0.460634 | -0.542122 | -0.454767 | -0.453329 | 0.194074 | 32/74 | False |
| RW11 | Three-step (feasible) | 70 | 70 | -1.155865 | -0.560257 | -0.485940 | -0.543244 | -0.463858 | -0.455597 | 0.142124 | 11/70 | True |
| RW11 | Pulsed (feasible) | 51 | 51 | -2.111547 | -1.500899 | -1.301455 | -1.257336 | -0.909432 | -0.473218 | 0.379753 | 1/51 | True |
| RW12 | CCCV (feasible) | 49 | 49 | -1.345216 | -0.783689 | -0.750152 | -0.798389 | -0.692983 | -0.662911 | 0.180946 | 1/49 | True |
| RW12 | Two-step (feasible) | 42 | 42 | -1.303867 | -0.805004 | -0.714167 | -0.786298 | -0.664409 | -0.642145 | 0.178987 | 3/42 | True |
| RW12 | Three-step (feasible) | 33 | 33 | -1.110140 | -0.924075 | -0.790757 | -0.817577 | -0.717583 | -0.662445 | 0.126077 | 2/33 | True |
| RW12 | Pulsed (feasible) | 49 | 49 | -2.207376 | -1.867548 | -1.544142 | -1.528641 | -1.194398 | -0.871107 | 0.381216 | 1/49 | True |

Top 6 feasible candidates per cell, ranked by loss (the rule GP-BO uses to pick a winner):

### RW9

| Rank | Family | Eval | Loss | Reward | Δreward from best feasible | Within 1% |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | cccv | 37 | 0.683409 | -0.656342 | 0.020808 | False |
| 2 | cccv | 29 | 0.683721 | -0.656171 | 0.020637 | False |
| 3 | cccv | 55 | 0.684934 | -0.652315 | 0.016781 | False |
| 4 | two_step | 32 | 0.684970 | -0.657270 | 0.021736 | False |
| 5 | two_step | 78 | 0.687451 | -0.659617 | 0.024083 | False |
| 6 | two_step | 72 | 0.688600 | -0.660700 | 0.025166 | False |

Feasible candidates within 1% of this cell's best feasible reward: **5 / 244** (best reward -0.635534).

### RW10

| Rank | Family | Eval | Loss | Reward | Δreward from best feasible | Within 1% |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | cccv | 33 | 0.612423 | -0.588306 | 0.000000 | True |
| 2 | three_step | 64 | 0.617845 | -0.593578 | 0.005272 | True |
| 3 | three_step | 73 | 0.617884 | -0.593617 | 0.005311 | True |
| 4 | three_step | 42 | 0.619582 | -0.595215 | 0.006909 | False |
| 5 | cccv | 61 | 0.619905 | -0.595288 | 0.006982 | False |
| 6 | three_step | 58 | 0.620668 | -0.596235 | 0.007928 | False |

Feasible candidates within 1% of this cell's best feasible reward: **3 / 264** (best reward -0.588306).

### RW11

| Rank | Family | Eval | Loss | Reward | Δreward from best feasible | Within 1% |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | cccv | 26 | 0.464493 | -0.449510 | 0.000000 | True |
| 2 | cccv | 54 | 0.464495 | -0.449512 | 0.000002 | True |
| 3 | cccv | 35 | 0.465284 | -0.450318 | 0.000808 | True |
| 4 | cccv | 80 | 0.466561 | -0.451527 | 0.002017 | True |
| 5 | cccv | 16 | 0.467076 | -0.452292 | 0.002782 | True |
| 6 | cccv | 43 | 0.467174 | -0.452224 | 0.002714 | True |

Feasible candidates within 1% of this cell's best feasible reward: **33 / 273** (best reward -0.449510).

### RW12

| Rank | Family | Eval | Loss | Reward | Δreward from best feasible | Within 1% |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | two_step | 23 | 0.663412 | -0.642145 | 0.000000 | True |
| 2 | two_step | 21 | 0.667865 | -0.646332 | 0.004187 | True |
| 3 | two_step | 79 | 0.668071 | -0.646521 | 0.004376 | True |
| 4 | two_step | 31 | 0.674552 | -0.652635 | 0.010491 | False |
| 5 | two_step | 80 | 0.674571 | -0.652654 | 0.010509 | False |
| 6 | two_step | 77 | 0.675124 | -0.653174 | 0.011030 | False |

Feasible candidates within 1% of this cell's best feasible reward: **3 / 173** (best reward -0.642145).

