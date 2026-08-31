#!/usr/bin/env python3
"""Generate patches/kgpfn/0001..0005 from repos/kgpfn at the pinned SHA.

KGPFN (HKUST-KnowComp/KGPFN @ af415c33) is not an ULTRA fork, but its ranking
primitives are ULTRA's verbatim: pfn/tasks.py::compute_ranking is the 1-based
pessimistic-tie comparison and pfn/tasks.py::strict_negative_mask is the strict
filter, both byte-level copies of ULTRA's. Two things still had to be
established from its code before its numbers can sit in the shared tables, and
both are handled by the patches this script generates:

1. THE FILTER GRAPH. script/test_kgpfn.py filters against
   inference-graph edges + test targets (+ inverses) on every dataset.
   ULTRA's protocol (repos/ultra script/run.py) additionally filters the
   validation targets on ILPC and Ingram graphs. On the other 26 inductive
   graphs the two filters coincide (KGPFN evaluates the head question as the
   tail question of the inverse relation, and its inverse-target edges make
   that exactly ULTRA's h-side filter). Hence the dual-column dump: `rank` is
   the shared definition under ULTRA's filter, `rank_native` is what KGPFN's
   own eval computed -- the KG-ICL convention.

2. THE EVAL ENTRY. The shipped __main__ evaluates a `fast_test`-sized random
   subsample of the test split (256 in its test.yaml) and raises NameError
   without that key; the paper (arXiv 2605.14907) reports the full split.
   0001 makes the full split the default, 0003 adds a per-graph zero-shot
   config whose every knob is a CLI flag.

Regenerate with:  python3 scripts/gen_kgpfn_dump_patch.py
repos/kgpfn stays pristine; the scratch tree is throwaway.
"""
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("SCRATCH", "/tmp/kgfm-kgpfn-patchgen")
OUT = os.path.join(ROOT, "patches", "kgpfn")


def run(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout


def edit(path, pairs):
    full = os.path.join(SCRATCH, path)
    text = open(full).read()
    for old, new in pairs:
        assert text.count(old) == 1, "anchor {!r} x{} in {}".format(old[:70], text.count(old), path)
        text = text.replace(old, new)
    open(full, "w").write(text)


def write(path, text):
    full = os.path.join(SCRATCH, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as handle:
        handle.write(text)


def emit(number, name, reason):
    # intent-to-add so brand-new files (rank_dump.py, the eval config) appear
    # in `git diff`; without -N an untracked file yields an empty patch.
    run("git", "add", "-N", "-A", cwd=SCRATCH)
    diff = run("git", "diff", cwd=SCRATCH)
    assert diff.strip(), "empty diff for {}".format(name)
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "{}-{}.diff".format(number, name))
    with open(out_path, "w") as handle:
        handle.write("Reason: " + reason.strip() + "\n\n" + diff)
    run("git", "add", "-A", cwd=SCRATCH)
    run("git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", name, cwd=SCRATCH)
    print("wrote", out_path)


if os.path.isdir(SCRATCH):
    shutil.rmtree(SCRATCH)
shutil.copytree(os.path.join(ROOT, "repos", "kgpfn"), SCRATCH,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pth", "*.ckpt"))
# Upstream's .gitignore ignores config/ wholesale, which would silently drop
# the new eval config from `git diff`. The scratch git is throwaway; the
# ignore file is not part of any patch, so it goes before the base commit.
os.remove(os.path.join(SCRATCH, ".gitignore"))
run("git", "init", "-q", cwd=SCRATCH)
run("git", "add", "-A", cwd=SCRATCH)
run("git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", "base", cwd=SCRATCH)


# ---------------------------------------------------------------------------
# 0001 -- rank dump (dual definition) + full-test-split evaluation
# ---------------------------------------------------------------------------
write("pfn/rank_dump.py", '''"""Per-query rank dump for the shared ``ranks/`` schema, dual-definition.

This module writes ranks. It does not produce them and it does not compute a
metric: every native number it emits was already computed by
``script/test_kgpfn.py`` and is copied out unchanged, and the shared-definition
numbers are computed from the very same score tensor with the very same
``pfn/tasks.py`` functions, only against ULTRA's filter graph.

Why two columns. KGPFN ranks with ULTRA's own arithmetic --
``pfn/tasks.py::compute_ranking`` is 1-based with ``pos_pred <= pred``
(pessimistic ties) over a strict-negative mask that excludes the target, all
byte-identical to ULTRA -- so the tie rule and the offset match the shared
definition as-is. What does not always match is the FILTER GRAPH:

  * KGPFN filters against inference-graph edges + test targets (+ their
    inverse copies) on every dataset (``_make_filtered_data`` in
    script/test_kgpfn.py).
  * ULTRA's protocol (repos/ultra script/run.py, the "ILPC"/"Ingram" branch)
    additionally filters the VALIDATION targets on ILPC2022 and the three
    Ingram families -- 15 of the 41 inductive graphs.

On the other 26 graphs the two filters are the same set of (query, candidate)
pairs: KGPFN asks the head question ``(?, r, t)`` as the tail question
``(t, r^-1, ?)``, and the inverse-typed copies in its filter make its
``t_mask`` on the inverse query exactly ULTRA's ``h_mask`` on the original.
So:

  ``rank`` / ``n_candidates``                 shared definition (ULTRA filter);
                                              what every cross-model table reads.
  ``rank_native`` / ``n_candidates_native``   what KGPFN's own eval computed and
                                              what its CSVs are made of;
                                              criterion A reads this side.

Rows for the inverse queries are stored in base orientation with
``direction == "head"``: (h, r, t) is the original triple, and the rank
answers ``(?, r, t)`` -- the same meaning as in every other model's dump.

The query ordering is recovered rather than imposed, exactly as in the TRIX
dump (patches/trix/0001): a second ``DistributedSampler`` with the same
(length, num_replicas, rank) and untouched seed/epoch yields the identical
permutation the real DataLoader consumed, so a stable ``query_id`` can be
attached without altering the loader. Batches the upstream loop skips
("insufficient negatives" from the context builder) must call ``skip`` so the
cursor stays in step; skipped queries are then absent from this dump AND from
KGPFN's own metrics, which keeps the two sides comparable -- and ``write``
refuses a desynchronised cursor outright.
"""

import os

from torch.utils import data as torch_data

import suite  # shared/suite.py, on PYTHONPATH; the only dataset list in the project

#: Extra columns on top of suite.RANK_COLUMNS -- the KG-ICL dual-column
#: convention, plus the native candidate count because the native filter is
#: what KGPFN's own hits@10_50 consumed.
EXTRA = ("rank_native", "n_candidates_native")


class Dumper:
    """Accumulates one row per scored query and writes a single parquet file."""

    def __init__(self, spec, test_triplets, world_size, rank):
        if world_size != 1:
            raise RuntimeError(
                "rank dumping is single-process only (run with --gpus [0]); got "
                "world_size=%d. Under DDP each rank sees a shard and the ranks "
                "are summed together by all_reduce, which loses per-query "
                "identity." % world_size)
        self.spec = spec
        self.graph = suite.by_run_id(spec["dataset"])
        self.order = [int(i) for i in torch_data.DistributedSampler(test_triplets, world_size, rank)]
        self.cursor = 0
        self.skipped = 0
        self.columns = {name: [] for name in tuple(suite.RANK_COLUMNS) + EXTRA}

    def _take(self, size):
        query_ids = self.order[self.cursor:self.cursor + size]
        self.cursor += size
        assert len(query_ids) == size, "sampler order desynchronised from the loader"
        return query_ids

    def skip(self, size):
        """A batch the eval loop dropped still consumed sampler positions."""
        self._take(size)
        self.skipped += size

    def add(self, batch, shared_ranking, shared_num_negative, native_ranking, native_num_negative):
        """``batch`` is the loader's [B, 3] (h, t, r) block in base orientation.

        The rank tensors are length B (tail-only graphs) or 2B: the first B
        entries answer the tail question, the second B answer the head
        question via the inverse relation, in upstream's own concatenation
        order (``eval_batch = cat([batch, reverse_batch])``).
        """
        size = len(batch)
        query_ids = self._take(size)
        pos_h, pos_t, pos_r = [column.tolist() for column in batch.t()]

        directions = ("tail",) if len(native_ranking) == size else ("tail", "head")
        assert len(native_ranking) == size * len(directions)
        assert len(shared_ranking) == len(native_ranking)
        shared_ranking = shared_ranking.tolist()
        shared_num_negative = shared_num_negative.tolist()
        native_ranking = native_ranking.tolist()
        native_num_negative = native_num_negative.tolist()

        for d, direction in enumerate(directions):
            offset = d * size
            for i in range(size):
                self.columns["dataset"].append(self.graph.id)
                self.columns["model"].append(self.spec["model"])
                self.columns["seed"].append(self.spec["seed"])
                self.columns["direction"].append(direction)
                self.columns["query_id"].append(query_ids[i])
                self.columns["h"].append(pos_h[i])
                self.columns["r"].append(pos_r[i])
                self.columns["t"].append(pos_t[i])
                self.columns["rank"].append(shared_ranking[offset + i])
                self.columns["n_candidates"].append(shared_num_negative[offset + i])
                self.columns["rank_native"].append(native_ranking[offset + i])
                self.columns["n_candidates_native"].append(native_num_negative[offset + i])

    def write(self):
        import pyarrow
        import pyarrow.parquet

        assert self.cursor == len(self.order), (
            "dump saw %d of %d queries; a skipped batch must call skip()"
            % (self.cursor, len(self.order)))
        if self.skipped:
            print("WARNING: %d queries skipped by the eval loop are absent from "
                  "this dump AND from KGPFN's own metrics" % self.skipped)

        arrow_types = {"string": pyarrow.string(), "int64": pyarrow.int64()}
        names = tuple(suite.RANK_COLUMNS) + EXTRA
        schema = pyarrow.schema(
            [(n, arrow_types[suite.RANK_COLUMN_TYPES.get(n, "int64")]) for n in names])
        table = pyarrow.table({n: self.columns[n] for n in names}, schema=schema)

        os.makedirs(self.spec["dir"], exist_ok=True)
        path = os.path.join(self.spec["dir"], "%s.parquet" % self.graph.id.replace(":", "_"))
        pyarrow.parquet.write_table(table, path)
        return path
''')

edit("script/test_kgpfn.py", [
    (
        "from pfn import tasks, util\n",
        "from pfn import tasks, util\nfrom pfn.rank_dump import Dumper\n",
    ),
    (
        'def test(cfg, model, test_data, filtered_data=None, split: str = "valid"):\n',
        'def test(cfg, model, test_data, filtered_data=None, split: str = "valid", shared_filtered_data=None):\n',
    ),
    (
        "        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()\n"
        "        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)\n"
        "        test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)\n",

        "        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()\n"
        "        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)\n"
        "        test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)\n"
        "\n"
        "        # Rank dump only; every ranking KGPFN reports below is untouched.\n"
        "        # graph_name is e.g. 'FB15k237Inductive-v1'; the suite spells it\n"
        "        # 'FB15k237Inductive:v1', and suite.by_run_id resolves the rest.\n"
        "        dump_dir = os.environ.get(\"KGPFN_RANK_DUMP_DIR\")\n"
        "        dumper = None\n"
        "        if dump_dir and split == \"test\":\n"
        "            dumper = Dumper({\"dir\": dump_dir,\n"
        "                             \"dataset\": str(graph_name).replace(\"-\", \":\", 1),\n"
        "                             \"model\": \"kgpfn\",\n"
        "                             \"seed\": int(cfg.train.get(\"rank_dump_seed\", 0))},\n"
        "                            test_triplets, world_size, rank)\n",
    ),
    (
        "            if filtered_data is None:\n"
        "                t_mask, h_mask = tasks.strict_negative_mask(test_graph, eval_batch)\n"
        "            else:\n"
        "                t_mask, h_mask = tasks.strict_negative_mask(filters, eval_batch)\n",

        "            if filtered_data is None:\n"
        "                t_mask, h_mask = tasks.strict_negative_mask(test_graph, eval_batch)\n"
        "            else:\n"
        "                t_mask, h_mask = tasks.strict_negative_mask(filters, eval_batch)\n"
        "            # Shared-definition mask: ULTRA's filter graph. The list holds\n"
        "            # the SAME object as `filters` wherever the two filters\n"
        "            # coincide (all but ILPC/Ingram), so nothing is recomputed\n"
        "            # there -- see pfn/rank_dump.py for the derivation.\n"
        "            shared_t_mask = t_mask\n"
        "            if dumper is not None and shared_filtered_data is not None:\n"
        "                _shared_filters = shared_filtered_data[graph_idx]\n"
        "                if _shared_filters is not filters:\n"
        "                    shared_t_mask, _ = tasks.strict_negative_mask(_shared_filters, eval_batch)\n",
    ),
    (
        "            except RuntimeError as e:\n"
        "                logger.warning(f\"Skipping batch due to insufficient negatives: {e}\")\n"
        "                continue\n",

        "            except RuntimeError as e:\n"
        "                logger.warning(f\"Skipping batch due to insufficient negatives: {e}\")\n"
        "                if dumper is not None:\n"
        "                    dumper.skip(len(batch))\n"
        "                continue\n",
    ),
    (
        "            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)\n"
        "            num_t_negative = t_mask.sum(dim=-1)\n",

        "            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)\n"
        "            num_t_negative = t_mask.sum(dim=-1)\n"
        "\n"
        "            if dumper is not None:\n"
        "                if shared_t_mask is t_mask:\n"
        "                    shared_ranking, shared_num_negative = t_ranking, num_t_negative\n"
        "                else:\n"
        "                    shared_ranking = tasks.compute_ranking(t_pred, pos_t_index, shared_t_mask)\n"
        "                    shared_num_negative = shared_t_mask.sum(dim=-1)\n"
        "                dumper.add(batch, shared_ranking, shared_num_negative,\n"
        "                           t_ranking, num_t_negative)\n",
    ),
    (
        "        if rankings:\n"
        "            ranking = torch.cat(rankings)\n",

        "        if dumper is not None:\n"
        "            logger.warning(\"Rank dump written to %s\", dumper.write())\n"
        "\n"
        "        if rankings:\n"
        "            ranking = torch.cat(rankings)\n",
    ),
    (
        "    valid_filtered_data = _make_filtered_data(valid_data)\n"
        "    test_filtered_data = _make_filtered_data(test_data)\n",

        "    valid_filtered_data = _make_filtered_data(valid_data)\n"
        "    test_filtered_data = _make_filtered_data(test_data)\n"
        "\n"
        "    # ULTRA-protocol filter graphs for the rank dump (repos/ultra\n"
        "    # script/run.py): ILPC and Ingram graphs additionally filter the\n"
        "    # validation targets; on every other graph ULTRA's filter is the\n"
        "    # edge set _make_filtered_data already built, so the same object is\n"
        "    # reused and the dump recomputes nothing there. Naming trap: the\n"
        "    # variable `valid_data` above holds _data[2] -- the TEST split --\n"
        "    # and `test_data` holds _data[1], the validation split whose\n"
        "    # targets the ILPC/Ingram filter needs.\n"
        "    def _needs_valid_in_filter(name):\n"
        "        return any(str(name).startswith(p)\n"
        "                   for p in (\"ILPC2022\", \"FBIngram\", \"WKIngram\", \"NLIngram\"))\n"
        "\n"
        "    shared_filtered_data = []\n"
        "    for _test_g, _valid_g, _native in zip(valid_data, test_data, valid_filtered_data):\n"
        "        if not _needs_valid_in_filter(getattr(_test_g, \"dataset\", \"\")):\n"
        "            shared_filtered_data.append(_native)\n"
        "            continue\n"
        "        _inv_valid_index = _valid_g.target_edge_index.flip(0)\n"
        "        _inv_valid_type = _valid_g.target_edge_type + _test_g.num_relations // 2\n"
        "        shared_filtered_data.append(Data(\n"
        "            edge_index=torch.cat([_native.edge_index,\n"
        "                                  _valid_g.target_edge_index, _inv_valid_index], dim=1),\n"
        "            edge_type=torch.cat([_native.edge_type,\n"
        "                                 _valid_g.target_edge_type, _inv_valid_type]),\n"
        "            num_nodes=_test_g.num_nodes,\n"
        "        ).to(device))\n"
        "    cfg.train.rank_dump_seed = args.seed\n",
    ),
    (
        "    test(cfg, model, short_valid, filtered_data=valid_filtered_data, split=\"test\")\n",

        "    # Full held-out test split by default. As shipped this line evaluated\n"
        "    # `short_valid`, a fast_test-sized random subsample of the test split\n"
        "    # (256 in the stock test.yaml), and raised NameError when fast_test\n"
        "    # was absent. The paper (arXiv 2605.14907) reports the full split;\n"
        "    # setting fast_test restores the subsampled behaviour exactly.\n"
        "    eval_data = short_valid if \"fast_test\" in cfg.train else valid_data\n"
        "    test(cfg, model, eval_data, filtered_data=valid_filtered_data,\n"
        "         shared_filtered_data=shared_filtered_data, split=\"test\")\n",
    ),
])
emit("0001", "rank-dump", """\
emit one parquet row per scored query into the shared ranks/ schema, carrying BOTH rank definitions, and evaluate the full test split. KGPFN's compute_ranking and strict_negative_mask are byte-identical to ULTRA's (1-based, pessimistic ties, target excluded), so `rank_native` needs no translation of tie rule or offset -- but its filter graph omits the validation targets that ULTRA's protocol filters on ILPC/Ingram (15 of 41 graphs), so `rank` is recomputed from the same scores against ULTRA's filter, the dual-column convention KG-ICL established (patches/kg-icl/0001). On the other 26 graphs the filters provably coincide and the same mask object is reused. Head questions are evaluated upstream as inverse-relation tail questions; the dump stores them in base orientation so (h, r, t) means the same thing as in every other model's dump. The final eval call also changes: as shipped it scored a fast_test-sized random subsample of the test split and raised NameError without fast_test; the paper reports the full split, which is now the default, with fast_test restoring the subsample. Every ranking KGPFN itself reports is untouched.""")


# ---------------------------------------------------------------------------
# 0002 -- log what the checkpoint actually loaded
# ---------------------------------------------------------------------------
edit("script/test_kgpfn.py", [
    (
        "        model.load_state_dict(sd, strict=False)\n"
        "        print(\"Loaded model from checkpoint\")\n",

        "        _load_result = model.load_state_dict(sd, strict=False)\n"
        "        # strict=False is upstream's choice, kept -- but silently so: a\n"
        "        # checkpoint that matches nothing still \"loads\", and with\n"
        "        # init_model=false a missing key then runs on constructor-random\n"
        "        # weights. Print the evidence so the run log can prove the load.\n"
        "        print(\"Loaded model from checkpoint: %d tensors in file, %d missing keys, %d unexpected keys\"\n"
        "              % (len(sd), len(_load_result.missing_keys), len(_load_result.unexpected_keys)))\n"
        "        if _load_result.missing_keys:\n"
        "            print(\"  missing (first 10):\", _load_result.missing_keys[:10])\n"
        "        if _load_result.unexpected_keys:\n"
        "            print(\"  unexpected (first 10):\", _load_result.unexpected_keys[:10])\n",
    ),
])
emit("0002", "ckpt-load-log", """\
report missing/unexpected keys when loading the KGPFN checkpoint. Upstream loads with strict=False and prints a bare success line; with init_model=false (the zero-shot eval path this project runs) any key the checkpoint fails to cover silently keeps constructor-random weights, and nothing in the log would distinguish that from a clean load. The load itself is unchanged.""")


# ---------------------------------------------------------------------------
# 0003 -- per-graph zero-shot eval config
# ---------------------------------------------------------------------------
write("config/script/eval_zero_shot.yaml", '''# Zero-shot evaluation of ONE suite graph from a pretrained KGPFN checkpoint.
# Added by this project (patches/kgpfn/0003), not by upstream: the shipped
# config/script/test.yaml pins graphs: CoDExMedium, subsamples fast_test: 256
# test queries and re-initialises encoder/transformer weights from their
# separate checkpoints. None of that is what a benchmark run wants.
#
# Every {{ '{{ var }}' }} below becomes a REQUIRED CLI flag: pfn/util.py::parse_args
# builds the parser from the template variables it detects in this file.
# scripts/run_kgpfn.sh passes all of them on every invocation.
#
# The model block is verbatim from config/script/test.yaml at the pin; the
# checkpoint decides the weights (init_model: false), the block decides the
# architecture, and the two must agree -- patches/kgpfn/0002 prints the
# missing/unexpected-key evidence on every load.
output_dir: "{{ output_dir }}"
model_config_path: null

dataset:
  class: JointDataset
  graphs: "{{ dataset }}"
  root: "{{ data_root }}"

model:
  class: KGPFN
  structure_encoder: True
  enhance_structure: True
  structure_score_enhance: True
  with_relation: True
  context_graph: 0
  context_tail: false
  seq_chunk_size: 40
  feature_transformer: tabicl
  relation_model:
    class: RelNBFNet
    input_dim: 64
    hidden_dims: [64, 64, 64, 64, 64, 64]
    message_func: distmult
    aggregate_func: sum
    short_cut: yes
    layer_norm: yes
  entity_model:
    class: EntityNBFNet
    input_dim: 64
    entity_chunk_size: 64
    hidden_dims: [64, 64, 64, 64, 64, 64]
    message_func: distmult
    aggregate_func: sum
    short_cut: yes
    layer_norm: yes
  semantic_encoder:
    model_name: none
    dim: 384

task:
  name: MultiGraphPretraining
  num_negative: 256
  strict_negative: yes
  adversarial_temperature: 1.0
  # hits@10_50 on top of test.yaml's list: it is the one metric that consumes
  # the candidate count, so it proves the n_candidates side of the rank dump.
  metric: [mr, mrr, hits@1, hits@3, hits@10, hits@10_50]
  # In-context retrieval budget per query. The pinned repo's own test.yaml
  # ships 20/80 (and script/test_kgpfn.py forces 40 negatives on
  # NELLInductive:v1); the paper (arXiv 2605.14907) states 20/60. The pinned
  # code's values are the defaults here -- it is the code being measured --
  # and both are flags so the paper-config A/B costs one run.
  num_pos: {{ num_pos }}
  num_neg: {{ num_neg }}
  loss_weights: [1.0, 1.0]
  label_smoothing: 0.1
  context_label_correction: True

# Never constructed at num_epoch 0 (train_and_validate returns first); kept
# for shape parity with test.yaml.
optimizer:
  class: AdamW
  lr: 5.0e-5
  weight_decay: 1.0e-5

train:
  gpus: {{ gpus }}
  batch_size: {{ batch_size }}
  num_epoch: 0
  log_interval: 25
  batch_per_epoch: 100
  valid_eval_step_interval: 0
  # deliberately NO fast_test: the whole held-out test split is evaluated
  eval_chunk_size: 256
  eval_log_interval: 50
  max_checkpoints: 1
  checkpoint_dir: ./checkpoints/
  train_structure_encoder: false
  inverse_relation_semantic_mode: text
  init_model: false
  tabicl_config_path: "{{ tabicl_config }}"
  kgpfn_checkpoint: "{{ ckpt }}"
  use_wandb: false
''')
emit("0003", "eval-config", """\
add a per-graph zero-shot evaluation config. The shipped test.yaml hardcodes graphs: CoDExMedium, subsamples 256 test queries via fast_test and re-initialises weights from the separate encoder/transformer checkpoints; this config takes one suite graph, the data root, the KGPFN checkpoint and the context budget as CLI flags (util.parse_args turns every template variable into a required flag), evaluates the full test split and loads all weights from the single pretrained checkpoint (init_model: false). hits@10_50 is added to the metric list because it is the one metric that consumes the candidate count, proving the dump's n_candidates. num_pos/num_neg default to the repo's own 20/80 (the paper states 20/60 -- recorded in shared/published.json), exposed as flags so the A/B costs one run.""")


# ---------------------------------------------------------------------------
# 0004 -- the model's own metric CSV + cost accounting
# ---------------------------------------------------------------------------
edit("script/test_kgpfn.py", [
    (
        "import glob\nimport os\n",
        "import glob\nimport os\nimport time\n",
    ),
    (
        "    for graph_idx, (test_graph, filters) in enumerate(zip(test_data, filtered_data)):\n"
        "        graph_name = getattr(test_graph, \"dataset\", f\"graph_{graph_idx}\")\n"
        "        is_nell_inductive_v1 = graph_name == \"NELLInductive-v1\"\n",

        "    for graph_idx, (test_graph, filters) in enumerate(zip(test_data, filtered_data)):\n"
        "        graph_name = getattr(test_graph, \"dataset\", f\"graph_{graph_idx}\")\n"
        "        # Cost accounting: wall seconds and peak GPU memory per graph,\n"
        "        # measured around the WHOLE per-graph eval -- the in-context\n"
        "        # retrieval (build_context_relation_aware) included, because that\n"
        "        # retrieval is the method and its cost is a first-class result.\n"
        "        graph_t0 = time.time()\n"
        "        if torch.cuda.is_available():\n"
        "            torch.cuda.reset_peak_memory_stats()\n"
        "        is_nell_inductive_v1 = graph_name == \"NELLInductive-v1\"\n",
    ),
    (
        "            _write_dataset_csv(csv_path, dataset_records)\n"
        "            for k, v in graph_metrics.items():\n"
        "                collected_metric_values.setdefault(k, []).append(v)\n",

        "            _write_dataset_csv(csv_path, dataset_records)\n"
        "            for k, v in graph_metrics.items():\n"
        "                collected_metric_values.setdefault(k, []).append(v)\n"
        "\n"
        "            # The model's own metric values, in the shape ULTRA's\n"
        "            # run_many.py writes them: one CSV per graph, full float repr.\n"
        "            # Criterion A compares shared/metrics.py against what the\n"
        "            # model itself computed over the same ranks in the same\n"
        "            # process; upstream's metrics.csv is a stacked two-column\n"
        "            # sheet at 6 digits, which cannot serve that comparison.\n"
        "            # eval_seconds / peak_gpu_mem_bytes ride along as extra\n"
        "            # columns (harmless to the CSV reader): KGPFN's paper\n"
        "            # reports no cost, so this run's cost is part of the result.\n"
        "            results_dir = os.environ.get(\"KGPFN_RESULTS_DIR\")\n"
        "            if results_dir and split == \"test\":\n"
        "                os.makedirs(results_dir, exist_ok=True)\n"
        "                _row = {\"dataset\": str(graph_name).replace(\"-\", \":\", 1)}\n"
        "                _row.update({k: repr(v) for k, v in graph_metrics.items()})\n"
        "                _row[\"eval_seconds\"] = round(time.time() - graph_t0, 3)\n"
        "                if torch.cuda.is_available():\n"
        "                    _row[\"peak_gpu_mem_bytes\"] = torch.cuda.max_memory_allocated()\n"
        "                _out_path = os.path.join(\n"
        "                    results_dir,\n"
        "                    \"KGPFN_results_%s.csv\" % time.strftime(\"%Y-%m-%d-%H-%M-%S\"))\n"
        "                with open(_out_path, \"w\", newline=\"\", encoding=\"utf-8\") as _fh:\n"
        "                    _writer = csv.DictWriter(_fh, fieldnames=list(_row.keys()))\n"
        "                    _writer.writeheader()\n"
        "                    _writer.writerow(_row)\n"
        "                logger.warning(\"Own-metric csv written to %s\", _out_path)\n",
    ),
])
emit("0004", "results-csv", """\
record the metric values KGPFN itself computes, one ULTRA-shaped CSV per graph, so criterion A can be run against them -- upstream only writes a stacked metrics.csv at 6 printed digits, which cannot anchor a bitwise comparison. The dataset column carries the suite run_id spelling. eval_seconds and peak_gpu_mem_bytes ride along as extra columns: the whole per-graph eval is timed, in-context retrieval included, because KGPFN's paper reports no cost and cost is a first-class result here. Directory arrives via KGPFN_RESULTS_DIR (forwarded by scripts/docker_run.sh); unset means upstream behaviour exactly.""")


# ---------------------------------------------------------------------------
# 0005 -- seed numpy
# ---------------------------------------------------------------------------
edit("script/test_kgpfn.py", [
    (
        "import yaml\nfrom typing import Any\n\nimport torch\n",
        "import yaml\nfrom typing import Any\n\nimport numpy as np\nimport torch\n",
    ),
    (
        "    torch.manual_seed(args.seed + util.get_rank())\n",
        "    torch.manual_seed(args.seed + util.get_rank())\n"
        "    # Eval-time context retrieval draws through numpy as well as torch:\n"
        "    # pfn/tasks.py::build_context_relation_aware shuffles the 2-hop\n"
        "    # candidate list with np.random.shuffle. Upstream seeds only torch,\n"
        "    # so two runs of one graph sample different contexts and score\n"
        "    # differently. Same defect and same fix as FLOCK (patches/flock/0004).\n"
        "    np.random.seed(args.seed + util.get_rank())\n",
    ),
])
emit("0005", "seed-numpy", """\
seed numpy's global generator alongside torch's. KGPFN's eval-time in-context retrieval shuffles candidate lists with np.random.shuffle (pfn/tasks.py::build_context_relation_aware), and upstream seeds only torch -- so two runs of one graph from one checkpoint sample different contexts and produce different numbers. The seed value is args.seed + rank, exactly the pattern upstream uses for torch; nothing else changes. Same defect and same fix as FLOCK (patches/flock/0004).""")

print("all patches written to", OUT)
