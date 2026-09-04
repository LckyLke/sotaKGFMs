# SW1: the combination probe's two one-pass checkpoints on the benchmark (2026-09-04)

Stage SW1 of plan v18 (evaluation only): the round-wise swap (L1's rounds
0-2 and all heads, MX15's entity and relation steps of rounds 3-5;
`output/incite-combo/soups/part_late100_mx15.pth`) and the 0.5 parameter
soup of L1 and MX15 (`soup_L1_MX15_a50.pth`), both under
`configs/incite_phase1.yaml`, both from results/incite/COMBINATION_PROBE.md
(the CPU study of 4 Sep, where the swap was +0.0024 [+0.0009, +0.0039]
over L1 on held-out carved splits). Not recipe candidates. The probe's
expectation, written before this ran: near MX15's level, not +0.003
above it; the swap keeps L1's seen-answer cells and takes about a third
of the unseen-answer gain.

## Numbers (41 test graphs, seed 1024)

| model | ind_e MRR | ind_er MRR | dev, stratified |
| --- | --- | --- | --- |
| L1 | 0.4560 | 0.3852 | 0.3082 |
| MX15 | 0.4621 | 0.3893 | 0.3050 |
| swap (`ranks/incite-swap-l1mx15-last`) | 0.4580 | 0.3868 | 0.3065 |
| soup (`ranks/incite-soup-l1mx15-last`) | 0.4609 | 0.3880 | 0.3060 |

| pair | ind_e | ind_er |
| --- | --- | --- |
| swap − MX15 | −0.0041 [−0.0070, −0.0012], 5 of 18 | −0.0025 [−0.0062, +0.0009], 9 of 23 |
| swap − L1 | +0.0021 [−0.0003, +0.0045], 11 of 18 | +0.0016 [−0.0010, +0.0041], 16 of 23 |
| soup − MX15 | −0.0012 [−0.0026, +0.0001], 10 of 18 | −0.0013 [−0.0033, +0.0005], 9 of 23 |
| soup − L1 | +0.0049 [+0.0016, +0.0086], 13 of 18 | +0.0027 [+0.0005, +0.0055], 15 of 23 |

Per-scenario MRR:

| group | model | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- | --- |
| ind_e | L1 | 0.4929 | 0.2856 | 0.6221 | 0.3187 |
| ind_e | swap | 0.4938 | 0.2820 | 0.6227 | 0.3352 |
| ind_e | soup | 0.4882 | 0.2964 | 0.6188 | 0.3580 |
| ind_e | MX15 | 0.4856 | 0.3020 | 0.6132 | 0.3756 |
| ind_er | L1 | 0.4079 | 0.1724 | 0.5896 | 0.2032 |
| ind_er | swap | 0.4113 | 0.1697 | 0.5906 | 0.2141 |
| ind_er | soup | 0.4025 | 0.1841 | 0.5874 | 0.2352 |
| ind_er | MX15 | 0.4013 | 0.1892 | 0.5827 | 0.2518 |

## Reading

1. The expectation held exactly. The swap keeps L1's seen-answer cells
   (SQSA 0.4938 and UQSA 0.6227 against L1's 0.4929 and 0.6221) and takes
   a third of the unseen-answer gain (UQUA 0.3352 between L1's 0.3187 and
   MX15's 0.3756). Net +0.002 over L1 with no cell trade, and 0.004 below
   MX15 on ind_e.
2. The soup is MX15 at one pass with a milder profile: seen-answer cells
   above MX15's, unseen-answer cells below, +0.0049 / +0.0027 over L1
   with both intervals above zero, tied with MX15 (both intervals
   through zero). It sits 0.002 above the straight interpolation of its
   members' numbers, as on the dev suite.
3. Neither beats MX15 anywhere, so no combination of these two
   checkpoints is a recipe. The mechanism result stands: the mix's trade
   lives in the entity steps, and the early rounds carry the seen-answer
   evidence. The training design the probe pointed at (a per-node gate
   between two late-round message functions) has no evidence for it
   beyond L1's level and is not queued.
