# 05 — RW11 measured-data vs twin check

This folder is diagnostic only. Arm A / Phase 3 / Arm B GP-BO was **not** re-run.
The pre-registered C1–C3 verdict in `../02_armA/armA_verdict.json` is **unchanged** (FAIL).

NASA sign: recorded current I < 0 is charge. Charging samples are those with **I_chg = max(−I, 0) > 0** (the user request “I > 0” after converting to charging current). SOC is `aacopt.calibration_data._soc_trace` on `aacopt.nasa_data.load_steps` (Coulomb counting re-anchored at reference charge/discharge, present capacity interpolated between accepted reference measurements, OCV from `results/01_calibration_dataset/ocv_curves.npz`).

Bins: current [1.5, 2.5), [2.5, 3.5), [3.5, 4.62] A; SOC [0.50, 0.60), [0.60, 0.70), [0.70, 0.80), [0.80, 0.90]. A bin with n < 30 is **insufficient**.

## A) Model-free data

| Cell | I bin | SOC bin | n | status | V median | V IQR | T median | T IQR |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| RW9 | 1.5-2.5 A | 50-60% | 12720 | ok | 4.133 | 0.07125000000000004 | 29.28731 | 5.051389999999998 |
| RW9 | 1.5-2.5 A | 60-70% | 1203 | ok | 4.171 | 0.05550000000000033 | 27.35149 | 3.1691399999999987 |
| RW9 | 1.5-2.5 A | 70-80% | 51 | ok | 4.049 | 0.020000000000000462 | 22.26213 | 0.3980900000000034 |
| RW9 | 1.5-2.5 A | 80-90% | 40 | ok | 4.077500000000001 | 0.016499999999999737 | 22.785125 | 0.18343000000000131 |
| RW9 | 2.5-3.5 A | 50-60% | 363 | ok | 4.186 | 0.019000000000000128 | 32.97348 | 5.97297 |
| RW9 | 2.5-3.5 A | 60-70% | 10 | insufficient | 4.299 | 0.06125000000000025 | 30.73195 | 0.48695499999999825 |
| RW9 | 2.5-3.5 A | 70-80% | 0 | insufficient | nan | nan | nan | nan |
| RW9 | 2.5-3.5 A | 80-90% | 0 | insufficient | nan | nan | nan | nan |
| RW9 | 3.5-4.62 A | 50-60% | 111 | ok | 4.198 | 0.11999999999999922 | 29.52148 | 3.217419999999997 |
| RW9 | 3.5-4.62 A | 60-70% | 8 | insufficient | 4.4405 | 0.07850000000000001 | 30.623739999999998 | 3.666490000000003 |
| RW9 | 3.5-4.62 A | 70-80% | 4 | insufficient | 4.521 | 0.03275000000000006 | 25.407855 | 6.108002500000001 |
| RW9 | 3.5-4.62 A | 80-90% | 0 | insufficient | nan | nan | nan | nan |
| RW10 | 1.5-2.5 A | 50-60% | 12637 | ok | 4.14 | 0.07899999999999974 | 27.60521 | 5.427019999999999 |
| RW10 | 1.5-2.5 A | 60-70% | 902 | ok | 4.165 | 0.08374999999999932 | 25.64546 | 2.8579624999999993 |
| RW10 | 1.5-2.5 A | 70-80% | 10 | insufficient | 4.2 | 0.0 | 26.788645000000002 | 0.031109999999998195 |
| RW10 | 1.5-2.5 A | 80-90% | 0 | insufficient | nan | nan | nan | nan |
| RW10 | 2.5-3.5 A | 50-60% | 747 | ok | 4.182 | 0.03200000000000003 | 27.38746 | 1.9130900000000004 |
| RW10 | 2.5-3.5 A | 60-70% | 20 | insufficient | 4.1815 | 0.0495000000000001 | 27.37191 | 0.07776999999999745 |
| RW10 | 2.5-3.5 A | 70-80% | 2 | insufficient | 4.451 | 0.0 | 29.24611 | 0.007780000000000342 |
| RW10 | 2.5-3.5 A | 80-90% | 0 | insufficient | nan | nan | nan | nan |
| RW10 | 3.5-4.62 A | 50-60% | 85 | ok | 4.26 | 0.11799999999999944 | 26.85864 | 4.808679999999999 |
| RW10 | 3.5-4.62 A | 60-70% | 10 | insufficient | 4.4145 | 0.1852500000000008 | 30.01586 | 3.1570675 |
| RW10 | 3.5-4.62 A | 70-80% | 2 | insufficient | 4.5945 | 0.01249999999999929 | 28.725065 | 0.5132650000000041 |
| RW10 | 3.5-4.62 A | 80-90% | 0 | insufficient | nan | nan | nan | nan |
| RW11 | 1.5-2.5 A | 50-60% | 13476 | ok | 4.14 | 0.07399999999999984 | 24.52919 | 2.1011799999999994 |
| RW11 | 1.5-2.5 A | 60-70% | 1082 | ok | 4.166 | 0.07899999999999974 | 24.38912 | 1.4786099999999998 |
| RW11 | 1.5-2.5 A | 70-80% | 32 | ok | 4.2 | 0.0 | 24.77044 | 0.7237425000000002 |
| RW11 | 1.5-2.5 A | 80-90% | 0 | insufficient | nan | nan | nan | nan |
| RW11 | 2.5-3.5 A | 50-60% | 526 | ok | 4.183 | 0.022999999999999687 | 26.58368 | 2.7393199999999993 |
| RW11 | 2.5-3.5 A | 60-70% | 3 | insufficient | 4.383 | 0.13299999999999912 | 22.24124 | 1.4396950000000004 |
| RW11 | 2.5-3.5 A | 70-80% | 1 | insufficient | 4.471 | 0.0 | 19.08825 | 0.0 |
| RW11 | 2.5-3.5 A | 80-90% | 0 | insufficient | nan | nan | nan | nan |
| RW11 | 3.5-4.62 A | 50-60% | 128 | ok | 4.228 | 0.11425000000000018 | 25.04282 | 3.4980774999999973 |
| RW11 | 3.5-4.62 A | 60-70% | 8 | insufficient | 4.3260000000000005 | 0.3617499999999998 | 22.140075 | 0.8113624999999978 |
| RW11 | 3.5-4.62 A | 70-80% | 8 | insufficient | 4.533 | 0.10624999999999929 | 19.00179 | 1.545982500000001 |
| RW11 | 3.5-4.62 A | 80-90% | 1 | insufficient | 4.509 | 0.0 | 18.77386 | 0.0 |
| RW12 | 1.5-2.5 A | 50-60% | 5811 | ok | 4.153 | 0.05600000000000005 | 24.13541 | 2.78228 |
| RW12 | 1.5-2.5 A | 60-70% | 504 | ok | 4.2 | 0.020000000000000462 | 24.67166 | 1.9779050000000034 |
| RW12 | 1.5-2.5 A | 70-80% | 49 | ok | 4.154 | 0.03500000000000014 | 24.41519 | 0.4663000000000004 |
| RW12 | 1.5-2.5 A | 80-90% | 45 | ok | 4.2 | 0.0030000000000001137 | 24.97476 | 0.12434999999999974 |
| RW12 | 2.5-3.5 A | 50-60% | 67 | ok | 4.17 | 0.0680000000000005 | 25.58096 | 4.953849999999999 |
| RW12 | 2.5-3.5 A | 60-70% | 2 | insufficient | 4.2385 | 0.01249999999999929 | 25.61204 | 0.0 |
| RW12 | 2.5-3.5 A | 70-80% | 2 | insufficient | 4.52 | 0.001000000000000334 | 19.616245 | 0.1648349999999965 |
| RW12 | 2.5-3.5 A | 80-90% | 0 | insufficient | nan | nan | nan | nan |
| RW12 | 3.5-4.62 A | 50-60% | 47 | ok | 4.322 | 0.19599999999999973 | 24.83487 | 8.009989999999998 |
| RW12 | 3.5-4.62 A | 60-70% | 5 | insufficient | 4.594 | 0.07499999999999929 | 22.51889 | 2.4558699999999973 |
| RW12 | 3.5-4.62 A | 70-80% | 8 | insufficient | 4.6080000000000005 | 0.0912500000000005 | 19.34937 | 0.6632649999999991 |
| RW12 | 3.5-4.62 A | 80-90% | 2 | insufficient | 4.5969999999999995 | 0.03799999999999937 | 19.35722 | 0.0 |

### I_chg > 4 A and V ≥ 4.1 V (charging samples)

| Cell | n (I>4 A and V≥4.1 V) | n (I>4 A) | n (V≥4.1 V, charging) | max V at I>4 A |
| --- | ---: | ---: | ---: | ---: |
| RW9 | 41323 | 55833 | 1225280 | 4.663 |
| RW10 | 41066 | 54979 | 1214116 | 4.657 |
| RW11 | 44859 | 59923 | 1213623 | 4.648 |
| RW12 | 21904 | 25342 | 1166857 | 4.814 |

Figure: `figures/measured_V_vs_SOC_by_current_bin.png`.

SOC 80–90% median voltages by current bin:
- 1.5-2.5 A, SOC 80–90%: RW9 Vmed=4.077500000000001 (n=40); RW10=insufficient(n=0); RW11=insufficient(n=0); RW12 Vmed=4.2 (n=45)
- 2.5-3.5 A, SOC 80–90%: RW9=insufficient(n=0); RW10=insufficient(n=0); RW11=insufficient(n=0); RW12=insufficient(n=0)
- 3.5-4.62 A, SOC 80–90%: RW9=insufficient(n=0); RW10=insufficient(n=0); RW11=insufficient(n=1); RW12=insufficient(n=2)

## B) Twin vs measured voltage (same bins)

Predictions: each cell twin and `twin_source_RW9.pt`, on author-style non-overlapping 150-sample windows built from the same concatenated raw record. Errors are pred − meas on charging samples in the bin.

| Cell | model | I bin | SOC bin | n | status | bias (pred−meas) V | RMSE V | under-predicts V? |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| RW9 | cell_twin | 1.5-2.5 A | 50-60% | 12719 | ok | -0.014111451409393374 | 0.024877890246026005 | True |
| RW9 | cell_twin | 1.5-2.5 A | 60-70% | 1203 | ok | -0.024833217099222295 | 0.045438936452171855 | True |
| RW9 | cell_twin | 1.5-2.5 A | 70-80% | 51 | ok | -0.10693419134850599 | 0.11167425408552666 | True |
| RW9 | cell_twin | 1.5-2.5 A | 80-90% | 40 | ok | -0.10213084921836854 | 0.11460624209509995 | True |
| RW9 | cell_twin | 2.5-3.5 A | 50-60% | 363 | ok | -0.018001001762621337 | 0.0247446354938543 | True |
| RW9 | cell_twin | 2.5-3.5 A | 60-70% | 10 | insufficient | -0.06936094779968265 | 0.09327943229654573 | False |
| RW9 | cell_twin | 2.5-3.5 A | 70-80% | 0 | insufficient | nan | nan | False |
| RW9 | cell_twin | 2.5-3.5 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW9 | cell_twin | 3.5-4.62 A | 50-60% | 111 | ok | -0.04940391306834179 | 0.05883644759393936 | True |
| RW9 | cell_twin | 3.5-4.62 A | 60-70% | 8 | insufficient | -0.11231042766571064 | 0.1276818697974872 | False |
| RW9 | cell_twin | 3.5-4.62 A | 70-80% | 4 | insufficient | -0.19237087154388433 | 0.19337824681090637 | False |
| RW9 | cell_twin | 3.5-4.62 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW9 | twin_source_RW9.pt | 1.5-2.5 A | 50-60% | 12719 | ok | -0.014111451409393374 | 0.024877890246026005 | True |
| RW9 | twin_source_RW9.pt | 1.5-2.5 A | 60-70% | 1203 | ok | -0.024833217099222295 | 0.045438936452171855 | True |
| RW9 | twin_source_RW9.pt | 1.5-2.5 A | 70-80% | 51 | ok | -0.10693419134850599 | 0.11167425408552666 | True |
| RW9 | twin_source_RW9.pt | 1.5-2.5 A | 80-90% | 40 | ok | -0.10213084921836854 | 0.11460624209509995 | True |
| RW9 | twin_source_RW9.pt | 2.5-3.5 A | 50-60% | 363 | ok | -0.018001001762621337 | 0.0247446354938543 | True |
| RW9 | twin_source_RW9.pt | 2.5-3.5 A | 60-70% | 10 | insufficient | -0.06936094779968265 | 0.09327943229654573 | False |
| RW9 | twin_source_RW9.pt | 2.5-3.5 A | 70-80% | 0 | insufficient | nan | nan | False |
| RW9 | twin_source_RW9.pt | 2.5-3.5 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW9 | twin_source_RW9.pt | 3.5-4.62 A | 50-60% | 111 | ok | -0.04940391306834179 | 0.05883644759393936 | True |
| RW9 | twin_source_RW9.pt | 3.5-4.62 A | 60-70% | 8 | insufficient | -0.11231042766571064 | 0.1276818697974872 | False |
| RW9 | twin_source_RW9.pt | 3.5-4.62 A | 70-80% | 4 | insufficient | -0.19237087154388433 | 0.19337824681090637 | False |
| RW9 | twin_source_RW9.pt | 3.5-4.62 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW10 | cell_twin | 1.5-2.5 A | 50-60% | 12637 | ok | -0.01685219447869539 | 0.028950598282167905 | True |
| RW10 | cell_twin | 1.5-2.5 A | 60-70% | 902 | ok | -0.022624212484930725 | 0.04861380684293749 | True |
| RW10 | cell_twin | 1.5-2.5 A | 70-80% | 10 | insufficient | 0.001254367828368963 | 0.0013382190613425223 | False |
| RW10 | cell_twin | 1.5-2.5 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW10 | cell_twin | 2.5-3.5 A | 50-60% | 747 | ok | -0.021859628886742453 | 0.0315928575667429 | True |
| RW10 | cell_twin | 2.5-3.5 A | 60-70% | 20 | insufficient | -0.19866432647705076 | 0.20598946867675563 | False |
| RW10 | cell_twin | 2.5-3.5 A | 70-80% | 2 | insufficient | -0.12114598846435509 | 0.12144175759971083 | False |
| RW10 | cell_twin | 2.5-3.5 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW10 | cell_twin | 3.5-4.62 A | 50-60% | 85 | ok | -0.02928875584321869 | 0.04300754405579147 | True |
| RW10 | cell_twin | 3.5-4.62 A | 60-70% | 10 | insufficient | -0.12849594078063964 | 0.17926889993641187 | False |
| RW10 | cell_twin | 3.5-4.62 A | 70-80% | 2 | insufficient | -0.17921870422363284 | 0.1798894750631741 | False |
| RW10 | cell_twin | 3.5-4.62 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW10 | twin_source_RW9.pt | 1.5-2.5 A | 50-60% | 12637 | ok | -0.014862471292901294 | 0.026222826818685995 | True |
| RW10 | twin_source_RW9.pt | 1.5-2.5 A | 60-70% | 902 | ok | -0.023840639406191584 | 0.048940365074026916 | True |
| RW10 | twin_source_RW9.pt | 1.5-2.5 A | 70-80% | 10 | insufficient | 0.0019896030426023613 | 0.0024383272717728144 | False |
| RW10 | twin_source_RW9.pt | 1.5-2.5 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW10 | twin_source_RW9.pt | 2.5-3.5 A | 50-60% | 747 | ok | -0.0362614866790363 | 0.04111620307774175 | True |
| RW10 | twin_source_RW9.pt | 2.5-3.5 A | 60-70% | 20 | insufficient | -0.2177826831817627 | 0.22281806168478255 | False |
| RW10 | twin_source_RW9.pt | 2.5-3.5 A | 70-80% | 2 | insufficient | -0.1575527439117428 | 0.15769130812342944 | False |
| RW10 | twin_source_RW9.pt | 2.5-3.5 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW10 | twin_source_RW9.pt | 3.5-4.62 A | 50-60% | 85 | ok | -0.03496835201487827 | 0.042976527524375294 | True |
| RW10 | twin_source_RW9.pt | 3.5-4.62 A | 60-70% | 10 | insufficient | -0.12273746452331542 | 0.16787816059381827 | False |
| RW10 | twin_source_RW9.pt | 3.5-4.62 A | 70-80% | 2 | insufficient | -0.19342559051513675 | 0.19380807172138154 | False |
| RW10 | twin_source_RW9.pt | 3.5-4.62 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW11 | cell_twin | 1.5-2.5 A | 50-60% | 13476 | ok | -0.01529740851468239 | 0.03688221941893796 | True |
| RW11 | cell_twin | 1.5-2.5 A | 60-70% | 1082 | ok | -0.03519401268068837 | 0.06228818995901907 | True |
| RW11 | cell_twin | 1.5-2.5 A | 70-80% | 32 | ok | -0.016237803220749053 | 0.06277516608969394 | True |
| RW11 | cell_twin | 1.5-2.5 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW11 | cell_twin | 2.5-3.5 A | 50-60% | 526 | ok | -0.018043542959844214 | 0.028990302682312275 | True |
| RW11 | cell_twin | 2.5-3.5 A | 60-70% | 3 | insufficient | -0.08338943481445298 | 0.12528728928384356 | False |
| RW11 | cell_twin | 2.5-3.5 A | 70-80% | 1 | insufficient | -0.10028676605224618 | 0.10028676605224618 | False |
| RW11 | cell_twin | 2.5-3.5 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW11 | cell_twin | 3.5-4.62 A | 50-60% | 128 | ok | -0.03252301773428922 | 0.05853934489786152 | True |
| RW11 | cell_twin | 3.5-4.62 A | 60-70% | 8 | insufficient | -0.1737571525573729 | 0.2008666290482765 | False |
| RW11 | cell_twin | 3.5-4.62 A | 70-80% | 8 | insufficient | -0.13997997617721558 | 0.16254272128804428 | False |
| RW11 | cell_twin | 3.5-4.62 A | 80-90% | 1 | insufficient | -0.1801049079895023 | 0.1801049079895023 | False |
| RW11 | twin_source_RW9.pt | 1.5-2.5 A | 50-60% | 13476 | ok | -0.009112709806594555 | 0.027803931544348306 | True |
| RW11 | twin_source_RW9.pt | 1.5-2.5 A | 60-70% | 1082 | ok | -0.03255913327934566 | 0.06498737149093521 | True |
| RW11 | twin_source_RW9.pt | 1.5-2.5 A | 70-80% | 32 | ok | -0.02299825143814102 | 0.057140803309734145 | True |
| RW11 | twin_source_RW9.pt | 1.5-2.5 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW11 | twin_source_RW9.pt | 2.5-3.5 A | 50-60% | 526 | ok | -0.014360504875618248 | 0.026666596357199035 | True |
| RW11 | twin_source_RW9.pt | 2.5-3.5 A | 60-70% | 3 | insufficient | -0.11403877894083643 | 0.1511340254500211 | False |
| RW11 | twin_source_RW9.pt | 2.5-3.5 A | 70-80% | 1 | insufficient | -0.13171939468383798 | 0.13171939468383798 | False |
| RW11 | twin_source_RW9.pt | 2.5-3.5 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW11 | twin_source_RW9.pt | 3.5-4.62 A | 50-60% | 128 | ok | -0.009617232143878988 | 0.03415082784170642 | True |
| RW11 | twin_source_RW9.pt | 3.5-4.62 A | 60-70% | 8 | insufficient | -0.17466922283172592 | 0.20210832892997405 | False |
| RW11 | twin_source_RW9.pt | 3.5-4.62 A | 70-80% | 8 | insufficient | -0.16498561477661133 | 0.18289045198798787 | False |
| RW11 | twin_source_RW9.pt | 3.5-4.62 A | 80-90% | 1 | insufficient | -0.20402544403076206 | 0.20402544403076206 | False |
| RW12 | cell_twin | 1.5-2.5 A | 50-60% | 5810 | ok | -0.021900168371939224 | 0.04289050469954856 | True |
| RW12 | cell_twin | 1.5-2.5 A | 60-70% | 504 | ok | -0.037028918591756776 | 0.061731914160278956 | True |
| RW12 | cell_twin | 1.5-2.5 A | 70-80% | 49 | ok | -0.036214977342255227 | 0.08699216118744842 | True |
| RW12 | cell_twin | 1.5-2.5 A | 80-90% | 45 | ok | -0.014391467454698494 | 0.016277545554480063 | True |
| RW12 | cell_twin | 2.5-3.5 A | 50-60% | 67 | ok | -0.01981206330256678 | 0.07732183269023231 | True |
| RW12 | cell_twin | 2.5-3.5 A | 60-70% | 2 | insufficient | -0.09190809822082535 | 0.09298823173330069 | False |
| RW12 | cell_twin | 2.5-3.5 A | 70-80% | 2 | insufficient | -0.2091748905181885 | 0.2096957165518513 | False |
| RW12 | cell_twin | 2.5-3.5 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW12 | cell_twin | 3.5-4.62 A | 50-60% | 47 | ok | -0.10406485285657517 | 0.13697344059987052 | True |
| RW12 | cell_twin | 3.5-4.62 A | 60-70% | 5 | insufficient | -0.23859912643432626 | 0.2934215227479193 | False |
| RW12 | cell_twin | 3.5-4.62 A | 70-80% | 8 | insufficient | -0.2249739470481873 | 0.2327427899206505 | False |
| RW12 | cell_twin | 3.5-4.62 A | 80-90% | 2 | insufficient | -0.2422207565307617 | 0.2422272507445938 | False |
| RW12 | twin_source_RW9.pt | 1.5-2.5 A | 50-60% | 5810 | ok | 3.515353933257215e-05 | 0.038276854332411375 | False |
| RW12 | twin_source_RW9.pt | 1.5-2.5 A | 60-70% | 504 | ok | -0.014192639763393205 | 0.06977285466037386 | True |
| RW12 | twin_source_RW9.pt | 1.5-2.5 A | 70-80% | 49 | ok | -0.15067319282220332 | 0.15324345900207423 | True |
| RW12 | twin_source_RW9.pt | 1.5-2.5 A | 80-90% | 45 | ok | -0.10847604819403768 | 0.13701654359227583 | True |
| RW12 | twin_source_RW9.pt | 2.5-3.5 A | 50-60% | 67 | ok | -0.07727490501973168 | 0.09767580784365343 | True |
| RW12 | twin_source_RW9.pt | 2.5-3.5 A | 60-70% | 2 | insufficient | -0.199671865463257 | 0.20061991395977966 | False |
| RW12 | twin_source_RW9.pt | 2.5-3.5 A | 70-80% | 2 | insufficient | -0.22862197875976564 | 0.22863977938973712 | False |
| RW12 | twin_source_RW9.pt | 2.5-3.5 A | 80-90% | 0 | insufficient | nan | nan | False |
| RW12 | twin_source_RW9.pt | 3.5-4.62 A | 50-60% | 47 | ok | -0.05546308347012134 | 0.08903893951377014 | True |
| RW12 | twin_source_RW9.pt | 3.5-4.62 A | 60-70% | 5 | insufficient | -0.25178787002563485 | 0.2722687129439468 | False |
| RW12 | twin_source_RW9.pt | 3.5-4.62 A | 70-80% | 8 | insufficient | -0.2804802956581116 | 0.2852209620835147 | False |
| RW12 | twin_source_RW9.pt | 3.5-4.62 A | 80-90% | 2 | insufficient | -0.24722182464599607 | 0.24798732969263207 | False |

### Focused bin: 3.5–4.62 A × 80–90% SOC

| Cell | model | n | status | bias V | RMSE V | meas median V | pred median V |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| RW9 | cell_twin | 0 | insufficient | nan | nan | nan | nan |
| RW9 | twin_source_RW9.pt | 0 | insufficient | nan | nan | nan | nan |
| RW10 | cell_twin | 0 | insufficient | nan | nan | nan | nan |
| RW10 | twin_source_RW9.pt | 0 | insufficient | nan | nan | nan | nan |
| RW11 | cell_twin | 1 | insufficient | -0.1801049079895023 | 0.1801049079895023 | 4.509 | 4.328895092010498 |
| RW11 | twin_source_RW9.pt | 1 | insufficient | -0.20402544403076206 | 0.20402544403076206 | 4.509 | 4.304974555969238 |
| RW12 | cell_twin | 2 | insufficient | -0.2422207565307617 | 0.2422272507445938 | 4.5969999999999995 | 4.354779243469238 |
| RW12 | twin_source_RW9.pt | 2 | insufficient | -0.24722182464599607 | 0.24798732969263207 | 4.5969999999999995 | 4.349778175354004 |

The requested 3.5–4.62 A × 80–90% SOC bin has n<30 on every cell (RW11 n=1), so that specific question is **not identified** from data.

In the only well-populated high-current bin (3.5–4.62 A × 50–60% SOC, n≥30 on all four cells) the RW11 **cell twin under-predicts** voltage: bias = `-0.03252301773428922` V, RMSE = `0.05853934489786152` V (n=128). `twin_source_RW9.pt` on the same RW11 samples under-predicts less (bias = `-0.009617232143878988` V). All four cell twins under-predict in this bin.

## C) POST-HOC C3 margin (verdict unchanged)

Arm A C3 remains **FAIL**. This section only describes the failed (profile, twin) pairs.

Profiles that are not feasible on all four twins: 18 / 24.
Rejecting (profile, twin) pairs: 31.
Voltage rejections (V_peak > 4.20 + 1e-3): 31.
Rejecting pairs with overshoot_mV < that twin's held-out voltage RMSE: 26 / 31.
Voltage rejections with 0 < overshoot < twin RMSE: 26 / 31.

Twin RMSE (Phase 0 identical test windows, cell-specific checkpoint):
- RW9: 28.346536185207775 mV
- RW10: 32.32914213844151 mV
- RW11: 48.25838049076362 mV
- RW12: 63.92941710730615 mV

| profile | rejecting twin | V_peak | overshoot mV | voltage reject? | end_reason | twin RMSE mV | overshoot < RMSE |
| --- | --- | ---: | ---: | --- | --- | ---: | --- |
| RW9-P2 | RW10 | 4.207738876342773 | 7.73887634277326 | True | energy target | 32.32914213844151 | True |
| RW9-P2 | RW12 | 4.230576992034912 | 30.57699203491193 | True | energy target | 63.92941710730615 | True |
| RW9-P4 | RW10 | 4.207738876342773 | 7.73887634277326 | True | energy target | 32.32914213844151 | True |
| RW9-P4 | RW12 | 4.230576992034912 | 30.57699203491193 | True | energy target | 63.92941710730615 | True |
| RW10-P1 | RW9 | 4.216594219207764 | 16.594219207763494 | True | time budget | 28.346536185207775 | True |
| RW10-P2 | RW9 | 4.214328765869141 | 14.328765869140447 | True | time budget | 28.346536185207775 | True |
| RW10-P3 | RW9 | 4.215986728668213 | 15.986728668212713 | True | time budget | 28.346536185207775 | True |
| RW10-P5 | RW9 | 4.209707260131836 | 9.70726013183576 | True | time budget | 28.346536185207775 | True |
| RW10-P6 | RW9 | 4.212700843811035 | 12.700843811034979 | True | time budget | 28.346536185207775 | True |
| RW11-P1 | RW10 | 4.207738876342773 | 7.73887634277326 | True | energy target | 32.32914213844151 | True |
| RW11-P1 | RW12 | 4.230576992034912 | 30.57699203491193 | True | energy target | 63.92941710730615 | True |
| RW11-P2 | RW10 | 4.207738876342773 | 7.73887634277326 | True | energy target | 32.32914213844151 | True |
| RW11-P2 | RW12 | 4.230576992034912 | 30.57699203491193 | True | energy target | 63.92941710730615 | True |
| RW11-P3 | RW10 | 4.2070136070251465 | 7.013607025146307 | True | energy target | 32.32914213844151 | True |
| RW11-P3 | RW12 | 4.229182720184326 | 29.182720184325994 | True | energy target | 63.92941710730615 | True |
| RW11-P4 | RW10 | 4.207738876342773 | 7.73887634277326 | True | energy target | 32.32914213844151 | True |
| RW11-P4 | RW12 | 4.230576992034912 | 30.57699203491193 | True | energy target | 63.92941710730615 | True |
| RW11-P5 | RW10 | 4.20612907409668 | 6.12907409667951 | True | energy target | 32.32914213844151 | True |
| RW11-P5 | RW12 | 4.2276611328125 | 27.661132812499822 | True | energy target | 63.92941710730615 | True |
| RW11-P6 | RW10 | 4.206995487213135 | 6.995487213134588 | True | energy target | 32.32914213844151 | True |
| RW11-P6 | RW12 | 4.229152202606201 | 29.152202606200994 | True | energy target | 63.92941710730615 | True |
| RW12-P1 | RW9 | 4.249388694763184 | 49.388694763183416 | True | time budget | 28.346536185207775 | False |
| RW12-P1 | RW10 | 4.214388370513916 | 14.388370513915838 | True | time budget | 32.32914213844151 | True |
| RW12-P2 | RW9 | 4.250919342041016 | 50.91934204101545 | True | time budget | 28.346536185207775 | False |
| RW12-P2 | RW10 | 4.215041160583496 | 15.041160583495916 | True | time budget | 32.32914213844151 | True |
| RW12-P4 | RW9 | 4.2492570877075195 | 49.257087707519354 | True | time budget | 28.346536185207775 | False |
| RW12-P4 | RW10 | 4.214991092681885 | 14.991092681884588 | True | time budget | 32.32914213844151 | True |
| RW12-P5 | RW9 | 4.245351791381836 | 45.35179138183576 | True | time budget | 28.346536185207775 | False |
| RW12-P5 | RW10 | 4.213545799255371 | 13.545799255370916 | True | time budget | 32.32914213844151 | True |
| RW12-P6 | RW9 | 4.2494611740112305 | 49.46117401123029 | True | time budget | 28.346536185207775 | False |
| RW12-P6 | RW10 | 4.214949131011963 | 14.949131011962713 | True | time budget | 32.32914213844151 | True |

## Is RW11's higher current tolerance in the data or only in the twin?

The NASA records do not contain enough charging samples at 3.5–4.62 A and 80–90% SOC (n = 0, 0, 1, 2 for RW9–RW12) to test whether RW11 stays below 4.20 V in that regime, so that comparison is **insufficient**. Where the high-current data *are* sufficient (3.5–4.62 A × 50–60% SOC), RW11’s measured median voltage is `4.228` V (n=128), which is **not** below RW9 (`4.198` V, n=111) and is only slightly below RW10 (`4.26` V) and RW12 (`4.322` V); RW11 also has the **most** charging samples with I_chg > 4 A and V ≥ 4.1 V (44859 vs 41323 / 41066 / 21904), with max V at I>4 A = `4.648` V, similar to RW9/RW10. The RW11 cell twin **does** under-predict voltage in that sufficient high-current bin (bias `−0.03252301773428922` V), more than `twin_source_RW9.pt` on the same RW11 windows (bias `−0.009617232143878988` V). RW11’s extra current headroom in GP-BO is therefore a twin (especially fine-tune) effect, not a pattern the measured high-current voltages support.
