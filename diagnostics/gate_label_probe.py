"""CPU probe (2026-09-04, before PG3 ran): can the gate learn proof
relevance from the synthetic labels alone, and does the label design
matter? Trunk frozen at the 4-graph checkpoint, gates at bias 2 trained
by the two-sided proof loss only (150 steps, 8 instances per step), three
label designs (any2: PG3's, two negatives from anywhere in the instance;
any8: eight; near2: eight negatives whose source lies within two hops of
the query head); scored on held-out instances by the AUC of the gate
product between a query's proof edges and the other edges of its
instance, per round. Result (results/incite/gate_label_probe/): all three
reach AUC 0.92-0.94 from round 2 on (round 0 carries no messages), the
designs are indistinguishable, PG3 stays as configured.

CONTAINER-ONLY, CPU. Usage from a scratch copy of the package:
  python diagnostics/gate_label_probe.py any2 150 8 /kgfm-src/results/incite/gate_label_probe/probe_any2.json
"""
import json, os, sys, time
import torch, yaml
from easydict import EasyDict
from incite import synth
from incite.run import build_model

VARIANT = sys.argv[1]            # near2 | any2 | any8
STEPS = int(sys.argv[2])
K = int(sys.argv[3])
OUT = sys.argv[4]
torch.set_num_threads(4)
cfg = EasyDict(yaml.safe_load(open("/kgfm-src/configs/incite_phase1_4g_synth30_gate3.yaml")))
model = build_model(cfg)
sd = torch.load("/kgfm-src/output/incite-pretrain-4g/incite_last.pth", map_location="cpu", weights_only=False)
sd = sd.get("model", sd)
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected and all(k.startswith("gates.") for k in missing)
for n, p in model.named_parameters():
    p.requires_grad_(n.startswith("gates."))
scfg = synth.synth_config(cfg)
NEG = {"near2": 8, "any2": 2, "any8": 8}[VARIANT]
NEG_W = float(scfg["proof_neg_weight"])


def components_and_near(union, queries, hops=2):
    """per row: the node set of its instance (component of the head) and
    the nodes within `hops` of the head."""
    n = int(union.num_nodes)
    ei = union.edge_index
    adj = [[] for _ in range(n)]
    for a, b in ei.t().tolist():
        adj[a].append(b)
    rows = []
    for i in range(queries.shape[0]):
        head = int(queries[i, 0, 0])
        dist = {head: 0}
        frontier = [head]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        comp = torch.zeros(n, dtype=torch.bool)
        comp[torch.tensor(list(dist), dtype=torch.long)] = True
        near = torch.zeros(n, dtype=torch.bool)
        near[torch.tensor([v for v, d in dist.items() if d <= hops], dtype=torch.long)] = True
        rows.append((comp, near))
    return rows


def hard_negatives(union, queries, rows_info, per_pos):
    """replace the union's negatives by edges whose SOURCE is within 2 hops
    of the head (and not a proof edge), per_pos per proof edge"""
    E2 = union.edge_index.shape[1]
    src = union.edge_index[0]
    nrows, nedges = [], []
    gen = torch.Generator().manual_seed(7)
    for i, (comp, near) in enumerate(rows_info):
        proof = union.proof_edges[union.proof_rows == i]
        is_proof = torch.zeros(E2, dtype=torch.bool)
        is_proof[proof] = True
        cand = (near[src] & ~is_proof).nonzero().flatten()
        want = per_pos * int(proof.numel())
        if cand.numel() > want:
            cand = cand[torch.randperm(cand.numel(), generator=gen)[:want]]
        nrows.append(torch.full((cand.numel(),), i, dtype=torch.long))
        nedges.append(cand)
    union.proof_neg_rows = torch.cat(nrows)
    union.proof_neg_edges = torch.cat(nedges)


def make_union(step, k, neg):
    gen = torch.Generator().manual_seed(int(scfg["seed"]) + step)
    insts = synth.generate_instances(dict(scfg, instances_per_step=k), gen, k)
    union, queries = synth.union_batch(insts, k, gen, isolate_relations=False, proof_neg_per_pos=neg)
    return union, queries


def gate_products(union, queries):
    """[rows, E2] gate products per round, from forward hooks"""
    got = {}
    hooks = [g.register_forward_hook(lambda m, i, o, kk=kk: got.__setitem__(kk, o)) for kk, g in enumerate(model.gates)]
    with torch.no_grad():
        model(union, queries)
    for h in hooks:
        h.remove()
    src, typ = union.edge_index[0], union.edge_type
    return {kk: gn[:, src] * gr[:, typ] for kk, (gn, gr) in got.items()}


def auc(pos, neg):
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    # Mann-Whitney: P(pos > neg) with ties at 0.5
    p = pos.view(-1, 1); n = neg.view(1, -1)
    return float(((p > n).float() + 0.5 * (p == n).float()).mean())


def evaluate(tag, n_unions=6, k=8):
    res = {r: {"all": [], "near": [], "unreached_share": []} for r in range(len(model.gates))}
    means = {r: {"pos": [], "neg_all": [], "neg_near": []} for r in range(len(model.gates))}
    for j in range(n_unions):
        union, queries = make_union(900000 + j, k, 0)
        info = components_and_near(union, queries)
        prods = gate_products(union, queries)
        E2 = union.edge_index.shape[1]
        src = union.edge_index[0]
        for i, (comp, near) in enumerate(info):
            proof = union.proof_edges[union.proof_rows == i]
            if proof.numel() == 0:
                continue
            is_proof = torch.zeros(E2, dtype=torch.bool); is_proof[proof] = True
            in_inst = comp[src]
            neg_all = (in_inst & ~is_proof).nonzero().flatten()
            neg_near = (near[src] & ~is_proof).nonzero().flatten()
            for r, pr in prods.items():
                a = auc(pr[i, proof], pr[i, neg_all]); b = auc(pr[i, proof], pr[i, neg_near])
                if a is not None: res[r]["all"].append(a)
                if b is not None: res[r]["near"].append(b)
                means[r]["pos"].append(float(pr[i, proof].mean()))
                means[r]["neg_all"].append(float(pr[i, neg_all].mean()))
                if neg_near.numel(): means[r]["neg_near"].append(float(pr[i, neg_near].mean()))
    out = {}
    for r in res:
        out[r] = {"auc_all": round(sum(res[r]["all"]) / len(res[r]["all"]), 4),
                  "auc_near": round(sum(res[r]["near"]) / max(1, len(res[r]["near"])), 4),
                  "n": len(res[r]["all"]),
                  "prod_pos": round(sum(means[r]["pos"]) / len(means[r]["pos"]), 4),
                  "prod_neg_all": round(sum(means[r]["neg_all"]) / len(means[r]["neg_all"]), 4),
                  "prod_neg_near": round(sum(means[r]["neg_near"]) / max(1, len(means[r]["neg_near"])), 4)}
    print(tag, json.dumps(out), flush=True)
    return out


t0 = time.perf_counter()
before = evaluate("before")
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
model.train()
trace = []
for step in range(1, STEPS + 1):
    union, queries = make_union(step, K, NEG)
    if VARIANT == "near2":
        hard_negatives(union, queries, components_and_near(union, queries), NEG)
    proof = (union.proof_rows, union.proof_edges, union.proof_neg_rows, union.proof_neg_edges, NEG_W)
    opt.zero_grad()
    _s, aux = model(union, queries, proof=proof, return_aux=True)
    aux.backward()
    opt.step()
    trace.append(round(float(aux), 4))
    if step % 25 == 0:
        print("step", step, "aux %.4f" % float(aux), "%.0fs" % (time.perf_counter() - t0), flush=True)
model.eval()
after = evaluate("after")
json.dump({"variant": VARIANT, "steps": STEPS, "k": K, "neg_per_pos": NEG, "neg_weight": NEG_W,
           "before": before, "after": after, "trace": trace,
           "seconds": round(time.perf_counter() - t0, 1)}, open(OUT, "w"), indent=1)
print("written", OUT, flush=True)
