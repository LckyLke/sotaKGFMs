# Changed hyperparameters vs the plan (plan rule 4)

| where | plan | as run | status |
| --- | --- | --- | --- |
| stage A readout lr | 1e-3 | 5e-4 | deviation, discovered post-hoc; stage A plateaued at 3k steps so lr was likely not binding. Config now carries lr_stage_a: 1e-3 for any rerun. |
| stage B encoder lr | 1e-4 | first attempt 5e-5 (scale 0.1) | **fixed**: scale 0.2 -> 1e-4; stage B restarted from the stage-A best at step 0 |
| stage A steps | 20000 | stopped at ~5300 | deliberate: validation flat since step 1000; best 0.4130 banked (EARLY_STOP.json) |
| stage B steps | TRIX's 800k | 20000 cap + plateau stop | deliberate: 800k = ~90 GPU-days is out of budget; plateau policy instead |
| eval candidate chunk | 4096 | 4096 | matches |
| seed | plan rev-2: 1024 | 1024 | matches |
