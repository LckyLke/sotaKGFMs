# FLOCK: interrupted by a host crash, resumable

State at the crash, 2026-08-27 ~13:40 local:

* **16 of 41 graphs complete**, parquets verified readable. All of GraIL
  (FB15k237/WN18RR/NELL v1-v4), HM 1k/3k/5k, ILPC2022:small.
* In flight, no output: **FBIngram:25**. The ind_er group after it is untouched.
* Stale claims cleared. TIMINGS.jsonl carries one duplicate FB15k237Inductive_v1
  row from the aborted first attempt (index 1, the JIT-cold one); keep the later
  row when analysing.

To resume (settings that produced the 16 -- do not change mid-run):

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLOCK_BATCH_DIVISOR=4 \
      FLOCK_WORKDIR=/kgfm-src/output/flock-run \
      scripts/docker_run.sh flock bash -c \
      '/kgfm-src/scripts/run_flock.sh ind_e "[0]"; /kgfm-src/scripts/run_flock.sh ind_er "[0]"'

Completed graphs are skipped via their parquets; no REDO flag.
Remaining: ~25 graphs. The two big ones (ILPC2022:large ~5h, HM:indigo ~5h)
dominate; the other 23 sum to roughly 11h at the measured x165 ULTRA ratio.
