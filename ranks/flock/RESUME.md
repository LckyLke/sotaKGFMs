# FLOCK: interrupted twice by host crashes, resumable

State after the second crash, 2026-08-28 morning:

* **17 of 41 graphs complete**, parquets verified readable. The 16 from the
  first crash plus **ILPC2022:large** (14,457 s = 4.0 h, 20,368 ranks,
  MRR 0.318, status ok in TIMINGS).
* **HM:indigo FAILED after 6.1 h** (started 01:00:47, failed 07:06,
  n128_ensemble16, test_batch_size 1, divisor 4). The error text is lost:
  the log lived in /tmp and the reboot cleared it. After this failure every
  remaining graph failed in ~1.9 s (wedged CUDA context cascade) and the
  loop still exited 0 -- the known exit-code trap. Only TIMINGS tells the
  truth.
* Working hypothesis: HM:indigo at n128 walks exhausts host RAM. FLOCK is
  CPU/RAM-bound, and both host crashes happened while a big FLOCK graph was
  in flight. Treat HM:indigo as the prime suspect for the crashes.
* One stale claim (ILPC2022_large) left; clear the whole claims dir before
  the next launch.

To resume the 23 normal graphs (settings that produced the 17 -- do not
change mid-run):

    rm -rf ranks/.claims-flock
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLOCK_BATCH_DIVISOR=4 \
      FLOCK_WORKDIR=/kgfm-src/output/flock-run \
      scripts/docker_run.sh flock bash -c \
      '/kgfm-src/scripts/run_flock.sh ind_e "[0]"; /kgfm-src/scripts/run_flock.sh ind_er "[0]"'

For HM:indigo, do NOT rerun with the same settings a third time. Run it
alone, last, with FLOCK_DATASETS=HM:indigo, FLOCK_BATCH_DIVISOR=8, a RAM
watch (`free -m` loop), and nothing else on the machine.

Completed graphs are skipped via their parquets; no REDO flag.
TIMINGS.jsonl carries duplicate rows from aborted attempts (status failed);
keep only status-ok rows when analysing.

Deprioritized 2026-08-28: INCITE implementation and pretraining run first
(user decision). Resume FLOCK when the GPU frees.
