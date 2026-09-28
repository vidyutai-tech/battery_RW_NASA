# Shared charging families under discrete current

Currents are restricted to 1.0, 1.5, 2.0, 3.0, and 4.5 A. Reward is the advantage over that cell's own 0.5C CCCV charge.

A family is listed as shared only when a feasible profile of that family exists on every cell, and a two-step or three-step is counted only when the later current is actually reached. The current can still differ by cell inside that shared family.

## 30% in 20 min

Shared families: CCCV.

| Family | RW9 | RW10 | RW11 | RW12 |
|---|---|---|---|---|
| CCCV | CCCV, 2.00 A, R=236 | CCCV, 2.00 A, R=246 | CCCV, 4.50 A, R=437 | CCCV, 2.00 A, R=207 |

Top distinct feasible profiles (up to 6), not six copies of one search point:

**RW9**

| Rank | Profile | Reward | Time (min) | Step reached |
|---:|---|---:|---:|---|
| 1 | CCCV, 2.00 A | 236 | 18.7 | yes |
| 2 | Three-step, 2.00/1.50/1.00 A | 207 | 19.6 | yes |
| 3 | Two-step, 2.00/1.50 A | 201 | 19.8 | yes |
| 4 | CCCV, 4.50 A | 193 | 18.9 | yes |

**RW10**

| Rank | Profile | Reward | Time (min) | Step reached |
|---:|---|---:|---:|---|
| 1 | CCCV, 2.00 A | 246 | 18.6 | yes |
| 2 | Two-step, 2.00/1.00 A | 246 | 18.6 | no, later current is not used |
| 3 | Two-step, 2.00/2.00 A | 246 | 18.6 | no, later current is not used |
| 4 | Two-step, 2.00/1.50 A | 246 | 18.6 | no, later current is not used |
| 5 | Three-step, 2.00/2.00/2.00 A | 246 | 18.6 | no, later current is not used |
| 6 | Three-step, 2.00/1.50/1.00 A | 246 | 18.6 | no, later current is not used |

**RW11**

| Rank | Profile | Reward | Time (min) | Step reached |
|---:|---|---:|---:|---|
| 1 | CCCV, 4.50 A | 437 | 10.7 | yes |
| 2 | CCCV, 3.00 A | 418 | 12.6 | yes |
| 3 | Two-step, 3.00/1.50 A | 418 | 12.6 | no, later current is not used |
| 4 | Two-step, 3.00/1.00 A | 418 | 12.6 | no, later current is not used |
| 5 | Two-step, 3.00/3.00 A | 418 | 12.6 | no, later current is not used |
| 6 | Two-step, 3.00/2.00 A | 418 | 12.6 | no, later current is not used |

**RW12**

| Rank | Profile | Reward | Time (min) | Step reached |
|---:|---|---:|---:|---|
| 1 | CCCV, 2.00 A | 207 | 18.6 | yes |
| 2 | Two-step, 2.00/1.00 A | 207 | 18.6 | no, later current is not used |
| 3 | Two-step, 2.00/2.00 A | 207 | 18.6 | no, later current is not used |
| 4 | Two-step, 2.00/1.50 A | 207 | 18.6 | no, later current is not used |
| 5 | Three-step, 2.00/2.00/2.00 A | 207 | 18.6 | no, later current is not used |
| 6 | Three-step, 2.00/1.50/1.50 A | 207 | 18.6 | no, later current is not used |

## 40% in 30 min

Shared families: CCCV, Two-step, Three-step.

| Family | RW9 | RW10 | RW11 | RW12 |
|---|---|---|---|---|
| CCCV | CCCV, 2.00 A, R=220 | CCCV, 2.00 A, R=326 | CCCV, 4.50 A, R=515 | CCCV, 2.00 A, R=278 |
| Two-step | Two-step, 2.00/1.50 A, R=217 | Two-step, 2.00/1.50 A, R=303 | Two-step, 3.00/2.00 A, R=393 | Two-step, 2.00/1.50 A, R=190 |
| Three-step | Three-step, 2.00/1.50/1.50 A, R=223 | Three-step, 2.00/1.50/1.50 A, R=313 | Three-step, 3.00/2.00/1.00 A, R=484 | Three-step, 2.00/1.50/1.50 A, R=266 |

Top distinct feasible profiles (up to 6), not six copies of one search point:

**RW9**

| Rank | Profile | Reward | Time (min) | Step reached |
|---:|---|---:|---:|---|
| 1 | Three-step, 2.00/1.50/1.50 A | 223 | 28.6 | yes |
| 2 | CCCV, 2.00 A | 220 | 28.0 | yes |
| 3 | Two-step, 2.00/1.50 A | 217 | 29.0 | yes |
| 4 | Pulsed, 3.00 A | 206 | 24.4 | yes |
| 5 | CCCV, 4.50 A | 148 | 30.0 | yes |

**RW10**

| Rank | Profile | Reward | Time (min) | Step reached |
|---:|---|---:|---:|---|
| 1 | CCCV, 2.00 A | 326 | 24.7 | yes |
| 2 | Two-step, 2.00/1.00 A | 326 | 24.7 | no, later current is not used |
| 3 | Two-step, 2.00/2.00 A | 326 | 24.7 | no, later current is not used |
| 4 | Two-step, 2.00/1.50 A | 326 | 24.7 | no, later current is not used |
| 5 | Three-step, 2.00/2.00/2.00 A | 326 | 24.7 | no, later current is not used |
| 6 | Three-step, 2.00/1.50/1.50 A | 313 | 25.1 | yes |

**RW11**

| Rank | Profile | Reward | Time (min) | Step reached |
|---:|---|---:|---:|---|
| 1 | CCCV, 4.50 A | 515 | 15.7 | yes |
| 2 | CCCV, 3.00 A | 511 | 16.8 | yes |
| 3 | Two-step, 3.00/1.50 A | 511 | 16.8 | no, later current is not used |
| 4 | Two-step, 3.00/1.00 A | 511 | 16.8 | no, later current is not used |
| 5 | Two-step, 3.00/3.00 A | 511 | 16.8 | no, later current is not used |
| 6 | Three-step, 3.00/3.00/1.00 A | 511 | 16.8 | no, later current is not used |

**RW12**

| Rank | Profile | Reward | Time (min) | Step reached |
|---:|---|---:|---:|---|
| 1 | CCCV, 2.00 A | 278 | 24.7 | yes |
| 2 | Two-step, 2.00/1.00 A | 278 | 24.7 | no, later current is not used |
| 3 | Two-step, 2.00/2.00 A | 278 | 24.7 | no, later current is not used |
| 4 | Two-step, 2.00/1.50 A | 278 | 24.7 | no, later current is not used |
| 5 | Three-step, 2.00/2.00/2.00 A | 278 | 24.7 | no, later current is not used |
| 6 | Three-step, 2.00/1.50/1.50 A | 266 | 25.1 | yes |

## 50% in 30 min

No family is feasible on more than one cell.

| Family | RW9 | RW10 | RW11 | RW12 |
|---|---|---|---|---|
| CCCV | none | none | CCCV, 3.00 A, R=513 | none |
| Two-step | none | none | Two-step, 3.00/1.50 A, R=446 | none |
| Three-step | none | none | Three-step, 3.00/3.00/1.00 A, R=513 | none |
| Pulsed | none | none | Pulsed, 3.00 A, R=352 | none |

Top distinct feasible profiles (up to 6), not six copies of one search point:

**RW9**

No feasible discrete profile.

**RW10**

No feasible discrete profile.

**RW11**

| Rank | Profile | Reward | Time (min) | Step reached |
|---:|---|---:|---:|---|
| 1 | CCCV, 3.00 A | 513 | 21.0 | yes |
| 2 | Two-step, 3.00/1.00 A | 513 | 21.0 | no, later current is not used |
| 3 | Two-step, 3.00/3.00 A | 513 | 21.0 | no, later current is not used |
| 4 | Two-step, 3.00/1.50 A | 513 | 21.0 | no, later current is not used |
| 5 | Two-step, 3.00/2.00 A | 513 | 21.0 | no, later current is not used |
| 6 | Three-step, 3.00/3.00/1.00 A | 513 | 21.0 | no, later current is not used |

**RW12**

No feasible discrete profile.

## 60% in 45 min

No family is feasible on all four cells. The widest overlaps are: CCCV on RW10, RW11, RW12; Two-step on RW10, RW11, RW12; Three-step on RW11, RW12; Pulsed on RW10, RW11.

| Family | RW9 | RW10 | RW11 | RW12 |
|---|---|---|---|---|
| CCCV | none | CCCV, 2.00 A, R=349 | CCCV, 3.00 A, R=607 | CCCV, 2.00 A, R=337 |
| Two-step | none | Two-step, 2.00/1.50 A, R=330 | Two-step, 3.00/1.00 A, R=597 | Two-step, 2.00/1.50 A, R=301 |
| Three-step | none | none | Three-step, 3.00/2.00/1.50 A, R=449 | Three-step, 2.00/1.50/1.00 A, R=247 |
| Pulsed | none | Pulsed, 3.00 A, R=111 | Pulsed, 3.00 A, R=387 | none |

Top distinct feasible profiles (up to 6), not six copies of one search point:

**RW9**

No feasible discrete profile.

**RW10**

| Rank | Profile | Reward | Time (min) | Step reached |
|---:|---|---:|---:|---|
| 1 | CCCV, 2.00 A | 349 | 38.3 | yes |
| 2 | Two-step, 2.00/1.50 A | 330 | 39.2 | yes |
| 3 | Pulsed, 3.00 A | 111 | 40.5 | yes |
| 4 | Pulsed, 3.00 A | 111 | 40.5 | yes |
| 5 | Pulsed, 3.00 A | 111 | 40.5 | yes |
| 6 | Pulsed, 3.00 A | 111 | 40.5 | yes |

**RW11**

| Rank | Profile | Reward | Time (min) | Step reached |
|---:|---|---:|---:|---|
| 1 | CCCV, 3.00 A | 607 | 27.6 | yes |
| 2 | CCCV, 4.50 A | 598 | 26.0 | yes |
| 3 | Two-step, 3.00/1.00 A | 597 | 25.2 | yes |
| 4 | Two-step, 3.00/3.00 A | 597 | 25.2 | no, later current is not used |
| 5 | Two-step, 3.00/1.50 A | 597 | 25.2 | no, later current is not used |
| 6 | Three-step, 3.00/3.00/1.00 A | 597 | 25.2 | no, later current is not used |

**RW12**

| Rank | Profile | Reward | Time (min) | Step reached |
|---:|---|---:|---:|---|
| 1 | CCCV, 2.00 A | 337 | 37.0 | yes |
| 2 | Two-step, 2.00/2.00 A | 337 | 37.0 | no, later current is not used |
| 3 | Two-step, 2.00/1.50 A | 337 | 37.0 | no, later current is not used |
| 4 | Three-step, 2.00/2.00/2.00 A | 337 | 37.0 | no, later current is not used |
| 5 | Two-step, 2.00/1.50 A | 301 | 38.4 | yes |
| 6 | Three-step, 2.00/1.50/1.00 A | 247 | 42.0 | yes |

