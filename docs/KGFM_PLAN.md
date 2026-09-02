# Research and Implementation Plan for a Prior-Fitted In-Context Knowledge Graph Foundation Model

Version 1.0, 2 September 2026. Prepared for execution by an autonomous coding agent.

## 1 Purpose of this document

This document specifies the research plan and the implementation tasks for a new knowledge graph foundation model (KGFM). It is written for an autonomous coding agent that will implement the code, run the experiments and write the phase reports. The plan is organized as a sequence of phases, and each phase ends at a decision gate. At a gate, the agent will stop, write the phase report and wait for confirmation before the next phase starts. This structure ensures that a failed hypothesis is detected early and at low cost.

All numbers quoted from the literature are reference values only. Before any comparison, the agent will reproduce the relevant number from a released checkpoint with the harness of Phase 1. This rule prevents claims that rest on unverified figures.

## 2 Background and reference values

A knowledge graph (KG) is a set of facts of the form (head entity, relation, tail entity). Entity prediction answers a query of the form (h, r, ?), and relation prediction answers a query of the form (h, ?, t). A KGFM must answer both query types on a graph that it never saw in training, that is, on a graph with new entities and new relations. Zero-shot inference means that no training on the target graph takes place. The two metrics are the mean reciprocal rank (MRR) and Hits@k, and higher values are better for both.

The standard evaluation suite is the collection of 57 graphs from the ULTRA paper. Most papers pretrain on three graphs, namely FB15k237, WN18RR and CoDExMedium, and evaluate zero-shot on the suite. FLOCK reports on 54 of these graphs, whereas KGPFN reports on all 57. Table 1 lists the reference values that this project must beat.

Table 1. Reference values for zero-shot inference.

| Model | Task | Graphs | MRR | Hits@10 | Hits@1 | Parameters | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ULTRA | entity | 54 | 0.366 | 0.518 | | 168,705 | FLOCK, Table 2 |
| TRIX | entity | 54 | 0.390 | 0.548 | | 87,138 | FLOCK, Table 2 |
| FLOCK | entity | 54 | 0.391 | 0.560 | | 801,969 | FLOCK, Table 2 |
| KGPFN | entity, with context | 57 | 0.432 | 0.628 | | not stated | KGPFN, unverified |
| ULTRA | relation | 54 | 0.724 | | 0.613 | 168,705 | FLOCK, Table 3 |
| TRIX | relation | 54 | 0.792 | | 0.687 | 87,138 | FLOCK, Table 3 |
| FLOCK | relation | 54 | 0.881 | | 0.817 | 801,969 | FLOCK, Table 3 |

Six lessons from the literature shape this plan.

1. Expressivity changes give small gains. TRIX adds 0.024 MRR over ULTRA, MOTIF adds 0.004, and FLOCK adds 0.001 over TRIX on entity prediction.
2. In-context examples give the largest gains. KG-ICL adds 0.046 MRR over ULTRA on 43 graphs, and KGPFN in zero-shot mode matches fine-tuned KG-ICL.
3. Relation prediction is the task with large open gains. FLOCK adds 0.089 MRR over TRIX and credits the joint update of entities and relations in one sequence encoder.
4. All recent models are weak on large dense graphs. FLOCK needs 27.9 GB per training batch, and its gain over ULTRA falls with graph density, with a correlation of minus 0.53.
5. No scaling study beyond eight pretraining graphs exists. ULTRA reported saturation after four graphs at 0.17 million parameters, whereas FLOCK reports gains with more graphs at 0.8 million parameters.
6. Evaluation practice is weak. KGFMs exploit half-links, that is, seen (h, r) or (r, t) pairs in the inference graph (arXiv 2606.18001). A 2026 review finds a significance test in only 2 of 25 backbone studies (MAKE, DOI 10.3390/make8070194).

Four results from adjacent fields motivate the central bet of this plan.

1. RDB-PFN (arXiv 2603.03805, ICML 2026) pretrains a relational in-context model on more than two million synthetic tasks from a relational prior generator.
2. PluRel (ICLR 2026 workshop) reports that synthetic data unlocks scaling laws for relational foundation models.
3. OpenRFM (arXiv 2606.04320) shows that the synthetic prior determines the in-context regime of such a model.
4. GraphPFN (arXiv 2509.21489) applies the same recipe to node-level graph tasks.

No published model applies this recipe to KG link prediction. This gap is the opportunity that the plan exploits.

## 3 Objective and hypotheses

The objective of this project is one pretrained model with three properties. It answers entity and relation queries zero-shot on any KG. It uses in-context examples of the query relation. It improves with the amount of synthetic and real pretraining data. Four hypotheses make this objective testable.

1. H1, the data hypothesis. A KGFM pretrained on synthetic KGs from a rule prior reaches ULTRA-level zero-shot accuracy with no real pretraining graph. A mixture of synthetic and real graphs improves with graph count where real graphs alone saturate.
2. H2, the context hypothesis. In-context examples of the query relation, combined with hard negatives, give larger gains than any change of the encoder. The gains are largest in the stratum without a seen half-link.
3. H3, the joint hypothesis. One network trained jointly on entity and relation prediction matches two specialized networks on both tasks.
4. H4, the measurement hypothesis. Reported KGFM gains shrink under family holdout, tuned trivial baselines and paired significance tests, whereas the gains of this project survive these controls.

The order of the phases follows the order of risk. The data hypothesis is tested first, because the model work is pointless if the prior fails.

## 4 Protocols, success criteria and kill rules

Two evaluation protocols will be used throughout the project.

1. Protocol S is the standard protocol. Pretraining uses FB15k237, WN18RR and CoDExMedium, and evaluation covers all 57 graphs. This protocol exists for comparability with the literature only.
2. Protocol H is the holdout protocol. The test families are split into two folds, defined in Appendix A. A model tested on one fold is pretrained on real graphs of the other fold and on synthetic graphs only. This protocol is the basis of every claim of true generalization.

The success criteria are fixed before the first run and apply to the final model.

1. The average zero-shot entity MRR exceeds the best baseline by at least 0.02 on the same graph set, at matched parameter count and matched pretraining graphs.
2. The model wins on at least 40 of 57 graphs against the best baseline, with a tie threshold of 0.005 MRR.
3. No family loses more than 0.05 MRR against the best baseline.
4. The average zero-shot relation MRR reaches at least 0.881 on the 54-graph set.
5. In the stratum without a seen half-link, the model exceeds the best baseline by at least 0.02 MRR.
6. On the five largest graphs, the MRR stays within 0.01 of ULTRA, and the latency per query stays within ten times the latency of ULTRA.
7. A paired Wilcoxon signed-rank test over graphs gives p below 0.01 against each baseline after Holm correction.

Five kill rules stop the work early, and each rule names the phase in which it is checked.

1. K0, reproduction, Phase 2. If a reproduced baseline deviates by more than 0.01 MRR from its paper on more than 10 graphs, all other work stops until the cause is found.
2. K1, scaling, Phase 4. The rule compares pretraining with 64 graphs against pretraining with 8 graphs, at 3 million and at 10 million parameters. If the gain is below 0.005 MRR at both sizes, the second part of H1 is rejected. Phase 5 then proceeds with 8 real graphs plus synthetic graphs, and only if K2 passed.
3. K2, prior, Phase 3. If a model pretrained on synthetic graphs only scores below 0.30 average zero-shot MRR on the real test graphs, the prior is wrong. Generator work then continues until the rule passes. The target value is 0.37.
4. K3, context, Phase 7. If the context ablation does not show an ordered drop from full context to shuffled context to no context, the context path is broken. Scaling stops until it is fixed.
5. K4, relation, Phase 5. If Model A scores below 0.80 relation MRR at the end of Phase 5, the walk encoder of Phase 6 becomes mandatory before any entity-prediction claim is made.

## 5 Repository layout and conventions

The repository will use the layout below.

```
kgfm/
  PLAN.md                     this document
  pyproject.toml
  kgfm/
    data/        registry.py  loaders.py  families.py  halflink.py  corpus.py
    harness/     evaluate.py  metrics.py  stats.py  trivial.py  report.py
    baselines/   ultra.py  trix.py  flock.py  kgpfn.py  motif.py
    synth/       schema.py  rules.py  entities.py  generate.py  fidelity.py  export.py
    models/      relation_encoder.py  entity_encoder.py  context.py  pfn.py
                 heads.py  walk_encoder.py  model_a.py
    train/       pretrain.py  sampler.py  negatives.py  losses.py
    diag/        petals.py  connecthub.py  rule_recovery.py  ablations.py
                 permutation.py  cost.py
  configs/       phase0/ ... phase7/
  scripts/       one shell script per experiment
  patches/       diffs against third-party code
  third_party/   ULTRA/  TRIX/  flock/  KGPFN/  MOTIF/  HYPER/
  data/          raw/  processed/  synthetic/  corpus/    (git-ignored)
  results/       <phase>/<experiment>/<seed>/
  reports/       phase0_report.md ... phase7_report.md
  tests/         pytest unit tests
```

The conventions below apply to every experiment.

1. Every run writes config.yaml, metrics.json, per_graph.csv, strata.csv, env.txt and log.txt into its results directory.
2. The file env.txt records the git hash of this repository and of every third-party repository, the Python version and the CUDA version.
3. Seeds 0, 1 and 2 are used for every number that enters a claim. Single-seed runs are screens only and are labelled as such in every table.
4. Third-party code is never edited in place. Changes go into patches/ as diff files with a one-line reason each.
5. Every phase report is written to reports/phaseN_report.md and lists the results file behind every number.
6. Checkpoints are selected on pretraining validation data only. Target validation data is never used for selection.
7. Every experiment first runs on the development subset of Phase 0 as a smoke test, and only then on the full suite.

## 6 Environment

The environment will be built once and recorded in env.txt.

1. Python 3.10 with PyTorch 2.3 or later and CUDA 12.
2. torch-geometric 2.5 or later with torch-scatter and torch-sparse builds for the same PyTorch version.
3. triton, which the HYPER relational sparse kernels require.
4. numpy, scipy, pandas, pyyaml, tqdm, networkx, scikit-learn, matplotlib and pytest.
5. Java 11 for AnyBURL, which serves as the rule-mining trivial baseline.

The third-party repositories will be cloned at pinned commits into third_party/, and every commit hash will be recorded in env.txt.

1. ULTRA, https://github.com/DeepGraphLearning/ULTRA, with the checkpoints ultra_3g.pth, ultra_4g.pth and ultra_50g.pth.
2. TRIX, https://github.com/yuchengz99/TRIX, with the entity and the relation checkpoints.
3. FLOCK, https://github.com/jw9730/flock, with the entity and the relation checkpoints and the PETALS generator.
4. KGPFN, https://github.com/HKUST-KnowComp/KGPFN, with its checkpoint if released.
5. MOTIF, https://github.com/HxyScotthuang/MOTIF, with the ConnectHub generator.
6. HYPER, https://github.com/HxyScotthuang/HYPER, for the Triton relational sparse matrix kernels.

Phases 0 to 4 need one GPU with 24 GB of memory or more. Phase 5 at 10 million parameters or more needs two to four GPUs with 40 GB or more. FLOCK inference over the full suite needs one GPU with 48 GB. Generator work in Phase 3 needs 8 CPU cores or more and no GPU.

## 7 Phase 0 Dataset registry and development subset

**Scope.** Phase 0 builds the dataset registry and the development subset that every later phase uses. The registry is the single source of truth for graph names, families, settings and sizes.

The tasks of Phase 0 are listed below.

1. The registry in kgfm/data/registry.py will wrap the ULTRA dataset loaders and will expose all 57 graphs by a canonical name.
2. Each entry will record the setting, namely transductive, inductive_e or inductive_er, the family and origin tags of Appendix A, and the sizes of the train graph and of the inference graph.
3. The sizes will include the entity count, the fact count, the relation count and the average degree. They will also include the density, defined as facts divided by entities, and the largest number of distinct relations at one entity.
4. The registry will store two graph lists, namely suite57.txt with all graphs and suite54.txt with the graphs that FLOCK reports on. The agent will derive the second list from the FLOCK tables and will document any ambiguity.
5. The development subset dev10.txt will contain FB15k237Inductive v1, WN18RRInductive v1, NELLInductive v1, ILPC2022 small, HM 1k, Metafam, FBNELL, WikiTopicsMT1 tax, the InGram WK-25 graph and CoDExSmall. This subset covers every setting and most families at small size.
6. The command `python -m kgfm.data.registry --check` will load every graph, print the statistics table and write it to results/phase0/registry.csv.

**Acceptance criteria.** Phase 0 is complete when the conditions below hold.

1. All 57 graphs load, and their sizes match the ULTRA repository within one percent.
2. Every graph carries a family tag and an origin tag.
3. The statistics table exists as CSV and as a markdown table in the phase report.

## 8 Phase 1 Evaluation harness

**Scope.** Phase 1 builds the evaluation harness that every later phase reports through. The harness implements the ranking protocol, the half-link strata, the family tags, the statistical tests, the trivial baselines and the report format. This harness is a deliverable on its own, because no unified KGFM harness exists in public.

### 8.1 Ranking protocol

The harness will follow the ULTRA evaluation exactly. For entity prediction, each test fact yields a tail query (h, r, ?) and a head query (?, r, t), where the head query uses the inverse relation. All entities of the inference graph are candidates, and the filtered setting removes every other true fact from the candidate list. For relation prediction, each test fact yields one query (h, ?, t), and all relations are candidates. MRR, Hits@1, Hits@3 and Hits@10 are reported per graph. This alignment ensures that the numbers are comparable with the published tables.

### 8.2 Half-link strata

A half-link is a seen (h, r, ·) pair or a seen (·, r, t) pair in the inference graph, where the inference graph excludes the test facts. The harness will assign each test query to one of four strata, which Appendix C defines formally. The strata are named both, head_only, tail_only and none. Metrics will be reported per stratum together with the stratum counts. The agent will compare these definitions with the definitions in arXiv 2606.18001 and will document every difference. This stratification is essential, because the stratum none is the only stratum that measures structural generalization without a memorized pair.

### 8.3 Family tags and leak flags

Every graph carries a family tag and an origin tag from Appendix A. Every run configuration lists its pretraining graphs. The harness will derive the pretraining families and will flag a test graph as leaked when its family or its origin appears in that list. All summary tables will then be reported for all graphs, for leaked graphs and for clean graphs. This split makes the effect of family leakage visible in every report.

### 8.4 Statistical procedures

Comparisons between two models will be paired over graphs, and the procedures below will be implemented in kgfm/harness/stats.py.

1. A Wilcoxon signed-rank test on the per-graph MRR differences.
2. A sign test with a tie threshold of 0.005 MRR, reported as wins, ties and losses.
3. A bootstrap confidence interval of the mean difference at 95 percent, with 10,000 resamples over graphs.
4. A Holm correction when one model is compared against several baselines.
5. The mean and the standard deviation over three pretraining seeds when several seeds exist.

Appendix C gives the exact procedures. These tests are required, because a difference of 0.01 MRR over 57 graphs is inside the seed noise of most models.

### 8.5 Trivial baselines

Two trivial baselines will be implemented, because the MAKE review shows that tuned trivial baselines erase many reported gains. The first baseline is a half-link heuristic. It scores a candidate tail by the logarithm of one plus the count of facts (·, r, t) in the inference graph. A weighted bonus is added when a path of length at most two links h and t. The second baseline is rule mining on the inference graph with AnyBURL, with rules of length at most three and maximum-confidence aggregation. Both baselines use only the inference graph of the target, so they are legitimate zero-shot methods. The bonus weight of the heuristic will be tuned on the validation split of each graph, and this tuning will be stated in every table.

### 8.6 Cost measurement

Every evaluation run will record the preprocessing time per graph, the latency per query and the peak GPU memory. The five largest graphs of the suite define the cost table, namely YAGO310, DBpedia100k, ConceptNet100k, FB15k237 and CoDExLarge. This table is required, because the weak point of every recent model is the large dense graph.

### 8.7 Report format

The command `python -m kgfm.harness.report <results_dir>` will produce summary.md with fixed tables. The tables are the overall averages by setting, by family and by stratum. Further tables give the comparison statistics against each baseline, the leak split, the seed statistics and the cost. The same numbers will be written as CSV files next to the markdown file. Every later phase report will embed these tables without modification.

### 8.8 Tests

Unit tests will cover the metric computation, the filtered ranking, the stratum assignment and the statistical procedures. The metric test will compare the harness against the ULTRA evaluation code on FB15k237Inductive v1 with the ultra_3g checkpoint, and the two results must agree to four decimals. The stratum test will use a hand-built graph of ten facts with one query per stratum.

**Acceptance criteria.** Phase 1 is complete when the conditions below hold.

1. The harness reproduces the published zero-shot MRR of the ultra_3g checkpoint within 0.005 on at least 50 of 57 graphs.
2. All unit tests pass.
3. A strata table and a leak table exist for the ULTRA run.
4. The phase report lists every graph where the deviation exceeds 0.005, together with a probable cause.

## 9 Phase 2 Baseline reproduction

**Scope.** Phase 2 runs every baseline from its released checkpoint on the full suite under Protocol S. The results are the reference rows of every later table.

The baseline runs are listed below.

1. ULTRA with the ultra_3g checkpoint on entity prediction. Relation prediction with ULTRA requires one pass per candidate relation and will be run in the way that TRIX describes.
2. TRIX with its entity checkpoint and with its relation checkpoint.
3. FLOCK with its entity checkpoint and with its relation checkpoint, with 16 passes as in the paper and with three inference seeds, because the walks are random.
4. KGPFN with its released checkpoint and its context, and a context-free run if the code permits it.
5. MOTIF with the three-path vocabulary if a checkpoint is released. Otherwise the run is skipped and the report says so.
6. The two trivial baselines of Phase 1.

Every run will record cost. The per-graph numbers will be compared with the published tables, and every deviation above 0.01 MRR will be listed in a discrepancy log with a probable cause. Kill rule K0 applies. The report will also list the intersection of the FLOCK graph set and the KGPFN graph set, so that later comparisons use one common set.

**Acceptance criteria.** Phase 2 is complete when the conditions below hold.

1. A per_graph.csv file exists for every baseline and for every task that the baseline supports.
2. The discrepancy log exists, and the K0 decision is recorded in the phase report.
3. The cost table exists for every baseline on the five largest graphs.
4. The common graph set is written to suite_common.txt.

## 10 Phase 3 Synthetic knowledge graph generator

**Scope.** Phase 3 builds the generator that produces synthetic KGs with known rules and controllable structure statistics. This generator is the core of hypothesis H1. Its design ports the relational prior generator of RDB-PFN to binary relations and adds what KGs need, namely logical rules, hub entities and controlled incompleteness. OpenRFM shows that the prior determines the in-context regime, so every hyper-prior below is tunable and will be fit in the fidelity loop of Section 10.2.

### 10.1 Generator components

The generator consists of seven components, and each component receives its own module in kgfm/synth/.

1. Schema sampler. The number of types T is sampled log-uniformly on [3, 40], and the number of relations R on [5, 300]. Each relation receives one to three domain types, one to three range types, and a cardinality class among one-to-one, one-to-many, many-to-one and many-to-many. A relation is symmetric with probability 0.1 and has an inverse partner with probability 0.2.
2. Rule sampler. A rule program over relation identifiers is sampled. The rule families are symmetry, inversion, implication, composition of length two and composition of length three. Two further families exist. A sibling rule has the form r1(x, y) and r1(x, z) imply r2(y, z). An exclusion rule has the form r1(x, y) implies not r2(x, y). Each rule receives a confidence from a Beta(5, 2) distribution. The number of rules is R multiplied by a factor sampled on [0.5, 3]. Exclusion rules create relation pairs with equal structure and opposite meaning, which is the PETALS situation.
3. Entity sampler. The number of entities N is sampled log-uniformly on [500, 50,000]. Each entity receives a type from a Dirichlet-distributed type population and a degree propensity from a Pareto distribution with a shape sampled on [1.5, 3]. The propensity creates hub entities.
4. Fact generation. Seed facts are sampled per relation under the domain, range, cardinality and propensity constraints. Forward chaining then applies the rule program for one to three rounds, and each derivation is accepted with the confidence of its rule as a Bernoulli probability. A noise fraction sampled on [0, 0.05] of random facts is added afterwards. Every derived fact keeps a record of the rule that produced it.
5. Incompleteness and queries. A fraction sampled on [0.2, 0.6] of the facts is removed. Removed facts with a derivation record form the query pool, and each query is tagged with its half-link stratum in the graph that remains. A parameter rho sets the fraction of queries in the stratum none. The removal step enforces this fraction, because it removes all facts of the query relation around the head or around the tail.
6. Context availability. Each query relation keeps at least 21 facts in the graph that remains, so that 20 context positives can be sampled. Relations below this count are excluded from the query pool.
7. Export. Each graph is written in the ULTRA text format with train, valid and test files, plus a JSON file with the schema, the rule program and the derivation records. A generator seed reproduces a graph exactly.

For pretraining, each synthetic graph is used in transductive mode with a train and a valid split of its facts. This choice matches the way in which ULTRA pretrains on FB15k237. For diagnostics, pairs of graphs from the same rule program with disjoint entities and renamed relations provide inductive splits.

### 10.2 Prior fidelity test

The fidelity test compares 1,000 synthetic graphs with the 57 real graphs on a fixed vector of statistics.

1. The size statistics, namely the entity count, the fact count, the relation count and the density.
2. The log-binned degree distribution and the Zipf slope of the relation frequencies.
3. The fraction of symmetric relations and the fraction of relation pairs with an inverse pattern.
4. The normalized edge counts of the ULTRA relation graph per edge type, namely head-to-head, head-to-tail, tail-to-head and tail-to-tail.
5. The graphlet spectrum of the ULTRA+ vocabulary V3, that is, the open and closed two-paths and three-paths, normalized by the fact count.
6. The largest number of distinct relations at one entity and the clustering coefficient of the undirected projection.

Three fidelity criteria will be checked.

1. For each statistic, the Kolmogorov-Smirnov distance between the synthetic distribution and the real distribution is at most 0.2.
2. For at least 80 percent of the statistics, every real graph lies within the 5th to 95th percentile range of the synthetic graphs.
3. A logistic regression on the standardized statistic vector cannot distinguish synthetic graphs from real graphs with an area under the curve above 0.75 in cross-validation.

The hyper-priors of the generator will be adjusted until all three criteria pass, and every adjustment will be logged with its effect. This loop is expected to take one week, and it is the scientific core of the phase. Importantly, the real graphs of the test fold are used for fidelity statistics only and never for training, so the loop does not leak labels.

### 10.3 Kill rule K2 experiment

The unchanged ULTRA architecture from third_party/ULTRA will be pretrained on synthetic graphs only, with 50, 200 and 1,000 graphs. Each model will be evaluated zero-shot on the 57 real graphs under Protocol S. A fourth run mixes the three standard real graphs with 1,000 synthetic graphs. Each configuration runs with one seed first and with three seeds when it passes 0.30 MRR. This result decides K2. A pass at 0.37 or above is the first publishable milestone, namely a KGFM without any real pretraining graph.

**Acceptance criteria.** Phase 3 is complete when the conditions below hold.

1. The generator produces at least 1,000 graphs per hour on 8 CPU cores.
2. Unit tests confirm correct rule application on a hand-built example, the absence of duplicate facts, disjoint splits and exact seed reproducibility.
3. The fidelity report shows the three criteria with their values.
4. The K2 table exists with three seeds for every passing configuration.
5. The gate decision is recorded in the phase report.

## 11 Phase 4 Pretraining corpus and scaling curves

**Scope.** Phase 4 tests the second part of hypothesis H1. This part states that accuracy improves with the number of pretraining graphs when the model is large enough and the data is diverse enough. Phase 4 also fixes the mixture that Phase 5 uses.

### 11.1 Corpus construction

Protocol H needs many real pretraining graphs whose families are disjoint from the test fold. The corpus module kgfm/data/corpus.py will assemble at least 64 real graphs per fold from three sources.

1. The train splits of the suite graphs in the pretraining fold.
2. Slices of large public KGs, namely Wikidata5M, ogbl-wikikg2, YAGO4, DBpedia, ConceptNet, full WordNet and NELL, in the fold that their origin permits. A slice is the k-hop neighborhood of a set of seed entities, capped at 100,000 facts, and slices from one source have disjoint entity sets.
3. New families that no test graph shares. These are event graphs (ICEWS14, ICEWS05-15 and GDELT slices with time dropped), biomedical graphs (PrimeKG, DRKG, OpenBioLink and ogbl-biokg) and scholarly slices (OpenAlex). Classic small graphs (UMLS, Kinship, Nations and Countries) and the document-derived graphs of GFM-RAG complete this group.

Every corpus graph carries a family tag, an origin tag, a license note and a size record. The document-derived graphs are tagged as noisy and enter the mixture study as a separate group. Biomedical graphs are used only for the fold whose test set does not contain Hetionet. This bookkeeping is required, because a leaked family invalidates a Protocol H claim.

### 11.2 Scaling curves

Four curves will be measured with the ULTRA architecture, and the TRIX architecture repeats the two decisive points.

1. Graph count. Nested pretraining sets of 4, 8, 16, 32, 64 and 128 graphs at 3 million parameters.
2. Parameter count. Models of 0.17, 1, 3, 10 and 30 million parameters at 64 graphs, where width and depth grow together as listed in Appendix B.
3. Synthetic share. Mixtures with 0, 25, 50, 75 and 100 percent synthetic graphs at 64 graphs in total.
4. Noisy share. Mixtures with the document-derived graphs at 0, 10 and 30 percent of the corpus at 64 graphs.

Every point runs with one seed as a screen. The points at 8 and 64 graphs, at 3 and 10 million parameters, run with three seeds, because they decide K1. Evaluation uses the test fold of Protocol H and the development subset. The report shows each curve as a table and as a plot, with per-stratum values for the decisive points. Importantly, ULTRA reported no gain above 0.17 million parameters, so the parameter curve also tests whether more graphs are required before larger models pay off.

**Acceptance criteria.** Phase 4 is complete when the conditions below hold.

1. The corpus contains at least 64 tagged graphs per fold, and the corpus manifest is written to data/corpus/manifest.csv.
2. The four curves exist as tables and plots in the phase report.
3. The K1 decision rests on three seeds and is recorded in the phase report.
4. The phase report names the mixture and the parameter budget for Phase 5, with the evidence behind the choice.

## 12 Phase 5 Model A

**Scope.** Phase 5 builds and trains Model A, the main model of the project. Model A combines four parts, namely a query-conditioned relation encoder, an entity encoder, a per-relation context set with hard negatives, and a prior-fitted transformer over the context. One joint head serves both tasks. The encoders are pluggable, and the first version uses message passing, because message passing is cheap and robust on dense graphs. The design follows KGPFN in shape and differs in three points, namely synthetic pretraining, joint training and the walk encoder option of Phase 6.

### 12.1 Components

The components below will be implemented in kgfm/models/.

1. Relation encoder. A message-passing network over the ULTRA relation graph with its four edge types, conditioned on the query relation through the labeling trick, with six layers. The HYPER relational sparse kernel performs the aggregation. A configuration flag switches to the TRIX relation graph with entity identity, which is more expressive and about ten times more expensive.
2. Entity encoder. An NBFNet with six layers over the inference graph, started at the head entity with the query relation vector, as in ULTRA. It produces one vector per entity.
3. Triple features. For a candidate triple (h, r, t), the feature vector concatenates x_h, x_t, z_r, the DistMult product, the TransE difference and the cosine similarity, as in KGPFN. An adapter network maps this vector to the token dimension.
4. Context set. The context of relation r contains 20 positive facts of r from the inference graph and 60 negative triples. The negatives are 20 random-tail corruptions, 20 same-relation corruptions and 20 structural corruptions. A same-relation corruption takes its tail from another fact of r. A structural corruption takes a two-hop neighbor of the head that has no r fact with it. The context is encoded once per relation and per graph and is cached. This reuse keeps the cost per query small. The query fact and its inverse are never part of the context.
5. Context transformer. A prior-fitted transformer with self-attention over the 80 context tokens and cross-attention from the query tokens to the context tokens. Each context token concatenates its triple feature with a label embedding. The default has four layers, dimension 256 and eight heads. All candidate tails of a query form the query tokens at once, so one cross-attention pass scores every candidate.
6. Joint head. A task token selects entity prediction or relation prediction. For entity prediction, the relation encoder is conditioned on r, and the entity encoder starts at h. For relation prediction, the TRIX initialization is used, where h receives a vector of ones, t receives a vector of minus ones and all relations receive ones. One pass then yields x_h, x_t and z for all relations, and each candidate relation forms a query token against its own cached context. A linear layer on the transformer output gives the score.

### 12.2 Context-free row

With zero context tokens, the transformer sees only the query token, and the model reduces to an encoder with a learned scorer. This configuration is the context-free row that every table must contain, because the baselines without context access are otherwise compared unfairly.

### 12.3 Training recipe

The training recipe is fixed below, and every deviation will be documented in the run configuration.

1. The pretraining corpus follows the protocol of the run, and the mixture follows the recommendation of Phase 4.
2. A batch holds one graph, one relation and 64 queries that share the context set of that relation.
3. The task is chosen per step with probability 0.5 for entity prediction and 0.5 for relation prediction.
4. Each entity query receives 32 negatives, namely 16 random tails, 8 same-relation tails and 8 self-adversarial tails that the current model scores highest among the wrong tails. This mixture follows the KMAS finding that random negatives give weak supervision.
5. The loss is the sum of a binary cross-entropy with adaptive weights on hard negatives and a softmax cross-entropy that ranks the positive above its negatives, as in KGPFN. Relation queries use a softmax over all relations.
6. The optimizer is AdamW with a learning rate of 0.0005, a cosine schedule, 5,000 warm-up steps, gradient clipping at 1.0 and bf16 precision.
7. Screens run for 100,000 steps, and final models run for 300,000 steps.
8. The parameter budgets are A-small at 1 million, A-base at 3 million, A-large at 10 million and A-xl at 30 million, with the widths of Appendix B.
9. Checkpoint selection uses pretraining validation data only.
10. The final model runs with three seeds under Protocol S and under Protocol H.

### 12.4 Inference

Inference has a per-graph part and a per-query part. The per-graph part builds the relation graph, encodes the context set of every relation and caches the results. The per-query part runs one encoder pass and one cross-attention pass. The cost table reports both parts separately on the five largest graphs and on ogbl-wikikg2 as a stress test, and an out-of-memory failure is reported as such.

### 12.5 Ablations

The ablations below will be run with A-base and one seed, and the three ablations with the largest effect will be repeated with three seeds.

1. No context, which is the context-free row.
2. Context positives only, without negatives.
3. Random negatives only, without the same-relation and structural negatives.
4. Mean-pooled context in place of the transformer.
5. Separate models per task in place of the joint head.
6. The TRIX relation graph in place of the ULTRA relation graph.
7. Entity encoder depth of 2, 3 and 6 layers.
8. Synthetic share of 0 and 50 percent at fixed corpus size.

**Acceptance criteria.** Phase 5 is complete when the conditions below hold.

1. A-base exists with three seeds under both protocols, and the harness tables exist with strata, leak split and statistics.
2. The ablation table exists.
3. The K4 check is recorded, and the K3 check of Phase 7 is scheduled.
4. The cost table exists, with the stress test result.

## 13 Phase 6 Walk encoder variant

**Scope.** Phase 6 adds the FLOCK walk encoder as a second encoder. Its purpose is the relation-prediction target of 0.881 and the sparse graphs, where FLOCK is strongest.

The walk encoder replaces the relation encoder and the entity encoder as one block. It samples non-backtracking random walks, anonymizes them with the FLOCK recording protocol, encodes them with a GRU and pools the proposals with the consensus rule. The outputs x and z feed the same triple features, context set and transformer as in Model A. The walk code will be imported from third_party/flock through a wrapper, and the walk count will follow the FLOCK size rule. The transformer is not used as the walk reader, because FLOCK showed a GRU advantage of 0.036 MRR at equal size. A density threshold, fit on pretraining graphs only, selects the encoder per graph at inference. This gate keeps the dense graphs on the message-passing path.

The experiments of Phase 6 are listed below.

1. The walk variant of A-base under Protocol S and Protocol H with three seeds.
2. Relation prediction with the walk variant, the message-passing variant and the density gate.
3. Ablations of walk count, walk length and the number of test passes.
4. A joint model against two separate models, which tests hypothesis H3 for the walk variant.

**Acceptance criteria.** Phase 6 is complete when the harness tables exist for every experiment and the relation-prediction result is compared with 0.881 in the phase report.

## 14 Phase 7 Diagnostics

**Scope.** Phase 7 tests the mechanism behind every number. A number alone does not confirm a mechanism, so each diagnostic below has a target.

1. PETALS. The generator from third_party/flock produces the instances, and the target is 100 percent accuracy.
2. ConnectHub. The generator from third_party/MOTIF produces the instances for k of 3, 4 and 5, and the target is 100 percent accuracy.
3. Rule recovery. Two hundred synthetic graphs are generated with rule families that were withheld from pretraining, namely composition of length three and sibling rules. MRR is reported for queries that only a withheld rule derives and for queries that a seen rule derives. ULTRA, TRIX and FLOCK are run on the same graphs.
4. Context ablations at inference. Four runs use full context, shuffled labels, context from a wrong relation and no context. The accuracy must fall in this order, and the gap between full context and no context must reach 0.02 MRR. This test decides K3.
5. Permutation test. Entity identifiers and relation identifiers are permuted three times on ten graphs. For the message-passing path, the per-query top-1 answer must be identical across permutations. For the walk path, the MRR difference across permutations must stay within 0.002, because the walks are random.
6. Half-link strata. The strata table from the harness is compared across all models, and the gain in the stratum none is reported with its confidence interval.
7. Cost. The cost table is reported for every model on the five largest graphs and on the stress test.

**Acceptance criteria.** Phase 7 is complete when every diagnostic has a result with a pass or fail against its target, and the K3 decision is recorded.

## 15 Phase 8 Fallback model B

**Scope.** Model B is the safe state of the art and runs only with a second worker or after Model A fails a gate. It starts from TRIX and adds five parts. These are the context module of Phase 5, the negatives of Section 12.3, joint training, and 16 or more pretraining graphs from the Phase 4 corpus. The fifth part is a SEMMA-style text channel that is active only when relation names exist. The expected gain is 0.01 to 0.03 MRR. Model B is evaluated with the same harness and the same protocols as Model A.

## 16 Schedule and compute budget

The schedule below assumes one engineer-agent and the hardware of Section 6. Phases 3 and 4 can overlap, because the generator runs on CPU while the scaling runs occupy the GPUs.

| Phase | Content | Duration | GPU budget |
| --- | --- | --- | --- |
| 0 | Registry and development subset | 3 days | none |
| 1 | Harness, tests, ULTRA reproduction | 1 week | 1 GPU-day |
| 2 | Baselines from checkpoints | 1 week | 5 GPU-days |
| 3 | Generator, fidelity loop, K2 runs | 3 weeks | 8 GPU-days |
| 4 | Corpus and scaling curves, K1 | 3 weeks | 40 GPU-days |
| 5 | Model A, ablations, K4 | 4 weeks | 60 GPU-days |
| 6 | Walk encoder variant | 2 weeks | 30 GPU-days |
| 7 | Diagnostics, K3 | 1 week | 5 GPU-days |
| 8 | Fallback model B, optional | 2 weeks | 20 GPU-days |

The first publishable milestone is the K2 pass in week six. The second milestone is the Phase 5 result in week thirteen. A phase that exceeds twice its duration stops, and the agent reports the cause and a revised estimate before any further work.

## 17 Reporting and decision gates

Every phase ends at a gate, and the gate decision is written in the phase report before any work of the next phase starts.

| Gate | After phase | Decision |
| --- | --- | --- |
| G0 | 0 | The registry matches the ULTRA repository. |
| G1 | 1 | The harness reproduces ULTRA within tolerance. |
| G2 | 2 | K0 passes, and the common graph set is fixed. |
| G3 | 3 | K2 passes, or the generator loop continues. |
| G4 | 4 | K1 decides the corpus and the budget for Phase 5. |
| G5 | 5 | K4 decides whether Phase 6 is mandatory. |
| G6 | 6 | The relation-prediction result is compared with 0.881. |
| G7 | 7 | K3 passes, and the final claims are listed with their evidence. |

Each phase report follows one template with five sections. These are the runs with their configuration hashes, the harness tables, and the discrepancies and failures. The last two sections are the gate decision with the kill-rule values and the next actions. Every number in a report links to a results file. This template makes the reports comparable and auditable.

## 18 Operating rules for the agent

The rules below apply to every phase.

1. The agent will never fabricate, estimate or extrapolate a number. A number that does not exist in a results file does not appear in a report.
2. The agent will reproduce a baseline before it compares against that baseline.
3. The agent will pin every version and will record every third-party commit hash in env.txt.
4. The agent will run every experiment on the development subset before the full suite.
5. The agent will use three seeds for every number that enters a claim, and it will label single-seed numbers as screens.
6. The agent will never select a checkpoint on target validation data.
7. The agent will write unit tests for the harness metrics, the stratum assignment, the statistical procedures and the generator invariants, and it will keep them green.
8. The agent will not edit third-party code in place, and it will record every patch in patches/ with a reason.
9. The agent will make every run resumable from its last checkpoint, and it will record every out-of-memory failure as a result.
10. The agent will stop at every gate and will wait for confirmation.
11. The agent will stop when a phase exceeds twice its planned duration and will report the cause.
12. The agent will keep the reports in plain markdown with CSV files next to every table.

## 19 References

The primary sources below are the basis of the reference values and of the design choices.

1. ULTRA. Galkin et al., Towards Foundation Models for Knowledge Graph Reasoning, ICLR 2024. https://arxiv.org/abs/2310.04562 and https://github.com/DeepGraphLearning/ULTRA
2. TRIX. Zhang et al., TRIX, Transferable Relation-Entity Interactions in crossing patterns. https://arxiv.org/abs/2502.19512 and https://github.com/yuchengz99/TRIX
3. MOTIF. Huang et al., How Expressive are Knowledge Graph Foundation Models, ICML 2025. https://arxiv.org/abs/2502.13339 and https://github.com/HxyScotthuang/MOTIF
4. KG-ICL. Cui et al., A Prompt-Based Knowledge Graph Foundation Model for Universal In-Context Reasoning, NeurIPS 2024. https://arxiv.org/abs/2410.12288 and https://github.com/nju-websoft/KG-ICL
5. FLOCK. Kim et al., Flock, A Knowledge Graph Foundation Model via Learning on Random Walks, ICLR 2026. https://arxiv.org/abs/2510.01510 and https://github.com/jw9730/flock
6. KGPFN. Gao et al., Unlocking the Potential of Knowledge Graph Foundation Model via In-Context Learning, 2026. https://arxiv.org/abs/2605.14907 and https://github.com/HKUST-KnowComp/KGPFN
7. Graphlets, the ULTRA+ paper. Amouzouvi et al., Graphlets as Building Blocks for Structural Vocabulary in Knowledge Graph Foundation Models, 2026. https://arxiv.org/abs/2605.06154
8. GAMMA. Xin et al., Geometric Structural Knowledge Graph Foundation Model, 2025. https://arxiv.org/abs/2512.22931
9. KMAS. Boosting Knowledge Graph Foundation Models via Enhanced Negative Sampling, 2026. https://arxiv.org/abs/2605.27023
10. Half-link study. Gregucci et al., Half a Link can Be Enough to Predict a Whole Link, 2026. https://arxiv.org/abs/2606.18001
11. Evaluation rigor review. Evaluation Rigor from GNNs to Graph Foundation Models, MAKE 2026. https://doi.org/10.3390/make8070194
12. HYPER. Huang et al., A Foundation Model for Inductive Link Prediction with Knowledge Hypergraphs, ICLR 2026. https://arxiv.org/abs/2506.12362 and https://github.com/HxyScotthuang/HYPER
13. SEMMA. Arun et al., A Semantic Aware Knowledge Graph Foundation Model, EMNLP 2025. https://arxiv.org/abs/2505.20422 and https://github.com/arvindh75/semma
14. RDB-PFN. Wang et al., Relational In-Context Learning via Synthetic Pre-training with Structural Prior, ICML 2026. https://arxiv.org/abs/2603.03805
15. OpenRFM. https://arxiv.org/abs/2606.04320
16. GraphPFN. Eremeev et al., A Prior-Data Fitted Graph Foundation Model, 2025. https://arxiv.org/abs/2509.21489
17. TabPFN. Hollmann et al., Accurate predictions on small data with a tabular foundation model, Nature 637, 2025. https://www.nature.com/articles/s41586-024-08328-6
18. GFM-RAG and G-reasoner. Luo et al., NeurIPS 2025 and ICLR 2026. https://arxiv.org/abs/2509.24276 and https://github.com/RManLuo/gfm-rag
19. KRLM. Zhuo et al., Knowledge Reasoning Language Model, ICLR 2026. https://arxiv.org/abs/2510.13909
20. TFMLinker. Liao et al., Universal Link Predictor by Graph In-Context Learning with Tabular Foundation Models, 2026. https://arxiv.org/abs/2602.08592
21. SIGIL. Yom Tov and Gal, Neural Message Passing on Structural Interaction Graphs for Fully-Inductive GNNs, 2026. https://arxiv.org/abs/2608.08567
22. PFN index. https://github.com/Cloudy1225/Awesome-Prior-Data-Fitted-Networks

## Appendix A Graph families and holdout folds

Table A1 assigns every graph of the suite to a family, an origin and a holdout fold. The fold assignment separates the Freebase origin from the Wikidata origin, because Wikidata imported Freebase content. Perfect separation is impossible, because YAGO, DBpedia and Wikidata all derive from Wikipedia. The origin tag records this overlap, and the leak split of the harness makes it visible. The fold table is a proposal, and the project owner confirms it before Phase 4 starts.

Table A1. Families, origins and folds.

| Graphs | Setting | Family | Origin | Fold |
| --- | --- | --- | --- | --- |
| `FB15k237`, `FB15k237_10`, `FB15k237_20`, `FB15k237_50` | transductive | Freebase | Freebase | A |
| `FB15k237Inductive:v1` to `v4` | inductive_e | Freebase | Freebase | A |
| InGram `FB-25`, `FB-50`, `FB-75`, `FB-100` | inductive_er | Freebase | Freebase | A |
| `HM:1k`, `HM:3k`, `HM:5k`, `HM:indigo` | inductive_e | Hamaguchi | Freebase, to be verified | A |
| `FBNELL` | inductive_er | Freebase and NELL | Freebase and NELL | A, leaked in both folds |
| `YAGO310` | transductive | YAGO | Wikipedia and WordNet | A |
| `DBpedia100k` | transductive | DBpedia | Wikipedia | A |
| `Metafam` | inductive_er | Metafam | synthetic | A |
| `ConceptNet100k` | transductive | ConceptNet | crowdsourced | A |
| `AristoV4` | transductive | Aristo | science text | A |
| `WN18RR` | transductive | WordNet | WordNet | B |
| `WN18RRInductive:v1` to `v4` | inductive_e | WordNet | WordNet | B |
| `NELL995`, `NELL23k` | transductive | NELL | NELL | B |
| `NELLInductive:v1` to `v4` | inductive_e | NELL | NELL | B |
| InGram `NL-0`, `NL-25`, `NL-50`, `NL-75`, `NL-100` | inductive_er | NELL | NELL | B |
| `CoDExSmall`, `CoDExMedium`, `CoDExLarge` | transductive | CoDEx | Wikidata | B |
| `WDsinger` | transductive | Wikidata | Wikidata | B |
| `ILPC2022:small`, `ILPC2022:large` | inductive_e | Wikidata | Wikidata | B |
| `WikiTopicsMT1` to `MT4`, eight graphs | inductive_er | Wikidata | Wikidata | B |
| InGram `WK-25`, `WK-50`, `WK-75`, `WK-100` | inductive_er | Wikidata | Wikidata | B |
| `Hetionet` | transductive | Hetionet | biomedical | B |

Fold A holds 22 graphs, and fold B holds 35 graphs. A model tested on fold A is pretrained on real graphs of fold B families plus synthetic graphs. A model tested on fold B is pretrained on real graphs of fold A families plus synthetic graphs. External corpus graphs follow the same origin rule, and the event, scholarly and classic families of Section 11.1 are permitted in both folds.

## Appendix B Configuration schemas

Every experiment is defined by one YAML file, and the schemas below are mandatory. Unknown keys raise an error.

Generator configuration.

```yaml
generator:
  seed: 0
  n_graphs: 1000
  types: {dist: loguniform, low: 3, high: 40}
  relations: {dist: loguniform, low: 5, high: 300}
  entities: {dist: loguniform, low: 500, high: 50000}
  degree_shape: {dist: uniform, low: 1.5, high: 3.0}
  p_symmetric: 0.1
  p_inverse: 0.2
  rule_factor: {dist: uniform, low: 0.5, high: 3.0}
  rule_families: [symmetry, inversion, implication, chain2, chain3, sibling, exclusion]
  confidence: {dist: beta, a: 5, b: 2}
  chaining_rounds: {dist: choice, values: [1, 2, 3]}
  noise_fraction: {dist: uniform, low: 0.0, high: 0.05}
  removal_fraction: {dist: uniform, low: 0.2, high: 0.6}
  rho_none: 0.3
  min_context_facts: 21
  export_format: ultra
```

Pretraining configuration.

```yaml
pretrain:
  protocol: H            # S or H
  test_fold: A           # A or B, used only under protocol H
  corpus:
    real_graphs: [...]   # names from data/corpus/manifest.csv
    synthetic_dir: data/synthetic/v1
    synthetic_share: 0.5
    noisy_share: 0.0
  model:
    encoder: mp          # mp or walk
    relation_graph: ultra   # ultra or trix
    budget: A-base       # A-small, A-base, A-large, A-xl
    context: {k_pos: 20, k_neg: 60, cache: true}
    pfn: {layers: 4, dim: 256, heads: 8}
    joint: true
  optim: {lr: 0.0005, warmup: 5000, steps: 300000, clip: 1.0, precision: bf16}
  negatives: {random: 16, same_relation: 8, self_adversarial: 8}
  seeds: [0, 1, 2]
  selection: pretrain_valid
```

Evaluation configuration.

```yaml
evaluate:
  suite: suite57         # suite57, suite54, suite_common or dev10
  tasks: [entity, relation]
  strata: true
  baselines: [ultra_3g, trix, flock, kgpfn, halflink, anyburl]
  stats: {wilcoxon: true, bootstrap: 10000, holm: true, tie: 0.005}
  cost: {graphs: [YAGO310, DBpedia100k, ConceptNet100k, FB15k237, CoDExLarge]}
```

Table B1 gives the width targets per parameter budget. The agent counts the parameters of each configuration and adjusts the widths until the count lies within ten percent of the budget.

Table B1. Parameter budgets.

| Budget | Encoder hidden size | Encoder layers | Transformer dimension | Transformer layers |
| --- | --- | --- | --- | --- |
| ULTRA default | 64 | 6 and 6 | none | none |
| A-small, 1 million | 96 | 6 and 6 | 128 | 2 |
| A-base, 3 million | 128 | 6 and 6 | 256 | 4 |
| A-large, 10 million | 256 | 8 and 8 | 384 | 6 |
| A-xl, 30 million | 384 | 8 and 8 | 640 | 8 |

## Appendix C Half-link strata and statistical procedures

Let G be the inference graph without the test facts, and let q be the test query with head h, relation r and tail t. The two indicator functions below define the strata.

```
head_seen(q) = 1 if there exists t' with (h, r, t') in G, else 0
tail_seen(q) = 1 if there exists h' with (h', r, t) in G, else 0
stratum(q)   = both      if head_seen and tail_seen
             = head_only if head_seen and not tail_seen
             = tail_only if tail_seen and not head_seen
             = none      otherwise
```

For a head query (?, r, t), the roles of the two indicators are swapped. The stratum counts are reported next to the metrics, because a stratum with fewer than 50 queries gives an unreliable average.

The statistical procedures operate on the vector d of per-graph MRR differences between model M and baseline B.

1. The Wilcoxon signed-rank test uses scipy.stats.wilcoxon on d with the two-sided alternative and reports the p-value.
2. The sign test counts wins as d above 0.005, losses as d below minus 0.005 and ties otherwise.
3. The bootstrap draws 10,000 resamples of the graphs with replacement, computes the mean of d for each resample and reports the 2.5th and the 97.5th percentiles.
4. The Holm correction orders the p-values of all baseline comparisons and adjusts them in the standard way.
5. When three seeds exist, d is computed per seed, and the mean and the standard deviation of the seed-level means are reported.

## Appendix D Generator pseudo-code

The pseudo-code below fixes the order of operations of the generator. Every random draw uses the generator seed, so that a graph is reproducible.

```
function generate_graph(config, seed):
    rng = RandomState(seed)
    schema = sample_schema(config, rng)          # types, relations, domains, ranges, cardinality
    rules  = sample_rules(schema, config, rng)   # rule program with confidences
    ents   = sample_entities(schema, config, rng)  # types and degree propensities
    facts  = sample_seed_facts(schema, ents, rng)
    derivations = {}
    for round in 1 .. config.chaining_rounds:
        new = apply_rules(rules, facts, rng)     # accept each derivation with rule confidence
        facts, derivations = merge(facts, new, derivations)
    facts = add_noise(facts, config.noise_fraction, rng)
    kept, removed = remove_facts(facts, config.removal_fraction, config.rho_none, rng)
    queries = [f for f in removed if f in derivations and context_available(f, kept, config)]
    queries = tag_strata(queries, kept)
    return export(kept, queries, schema, rules, derivations, seed)
```

The function remove_facts enforces rho_none. It removes all facts of the query relation around the head or around the tail of a chosen query. It repeats this step until the fraction of queries in the stratum none reaches rho_none. The function context_available checks that at least 21 facts of the query relation remain in the kept set. The export writes train.txt, valid.txt and test.txt in the ULTRA format and meta.json with the schema, the rules and the derivations.
