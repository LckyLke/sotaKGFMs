"""Training objective and batched steps.

The entity objective is TRIX's, reused VERBATIM from crest/train.py (crest
branch), which itself restates ``repos/trix/src/run_entity.py::
train_and_validate``: positives sampled from the training graph's target
edges, 512 strict negatives per query (``negative_sampling``), binary
cross-entropy with logits under self-adversarial negative weighting at
temperature 1, AdamW at 5e-4. ``edge_match``/``strict_negative_mask``/
``negative_sampling``/``self_adversarial_nll``/``sample_positive_triples``
below are byte-level copies of the crest module (which copied the first
three from ``trix/tasks.py`` at pin 7596e14e); copied, not imported, so the
tests exercise exactly what the container runs without needing TRIX -- and
a test pins ``self_adversarial_nll`` to the reference formula.

The relation objective (design D) is softmax cross-entropy over direct
relations from ``INCITE.forward_relation``; the joint loss is
``L_entity + lambda * L_relation`` with lambda from config.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

__all__ = ["edge_match", "strict_negative_mask", "negative_sampling",
           "sample_positive_triples", "self_adversarial_nll",
           "entity_batch_loss", "relation_batch_loss", "halflink_masks"]


# ---------------------------------------------------------------------------
# Verbatim copies (crest/train.py <- trix/tasks.py pin 7596e14e)
# ---------------------------------------------------------------------------
def edge_match(edge_index, query_index):
    base = edge_index.max(dim=1)[0] + 1
    from functools import reduce
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    num_match = end - start
    offset = num_match.cumsum(0) - num_match
    range_ = torch.arange(num_match.sum(), device=edge_index.device)
    range_ = range_ + (start - offset).repeat_interleave(num_match)
    return order[range_], num_match


def strict_negative_mask(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    query_index = torch.stack([pos_h_index, pos_r_index])
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    query_index = torch.stack([pos_t_index, pos_r_index])
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask


def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)


def sample_positive_triples(graph, batch_size: int,
                            generator: torch.Generator) -> torch.Tensor:
    """``[batch_size, 3]`` rows of (h, t, r), the shape TRIX's train loader
    yields. Target edges are the training split when the graph carries one
    (TRIX datasets do); the duck-typed toy graphs fall back to their direct
    edges, since their inverse half restates the same facts."""
    if getattr(graph, "target_edge_index", None) is not None:
        ei, et = graph.target_edge_index, graph.target_edge_type
    else:
        num_direct = int(graph.num_relations) // 2
        direct = (graph.edge_type < num_direct).nonzero(as_tuple=True)[0]
        ei, et = graph.edge_index[:, direct], graph.edge_type[direct]
    picks = torch.randint(et.shape[0], (batch_size,), generator=generator).to(ei.device)
    return torch.stack([ei[0, picks], ei[1, picks], et[picks]], dim=-1)


def self_adversarial_nll(pred: torch.Tensor, num_negative: int,
                         adversarial_temperature: float) -> torch.Tensor:
    """TRIX's loss, verbatim from run_entity.py::train_and_validate:
    BCE-with-logits over [positive | negatives], negatives weighted by a
    softmax of their own scores (self-adversarial) at ``adversarial_temperature``,
    or uniformly at temperature 0."""
    target = torch.zeros_like(pred)
    target[:, 0] = 1
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    neg_weight = torch.ones_like(pred)
    if adversarial_temperature > 0:
        with torch.no_grad():
            neg_weight[:, 1:] = F.softmax(pred[:, 1:] / adversarial_temperature, dim=-1)
    else:
        neg_weight[:, 1:] = 1 / num_negative
    loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
    return loss.mean()


# ---------------------------------------------------------------------------
# INCITE steps
# ---------------------------------------------------------------------------
def entity_loss_from_triples(model, graph, triples: torch.Tensor,
                             num_negative: int,
                             adversarial_temperature: float = 1.0,
                             strict: bool = True, support=None,
                             walk_offset: int = 0,
                             sampler=negative_sampling,
                             halflink=None) -> torch.Tensor:
    """TRIX's objective on given positives: sample negatives, score in one
    batched forward, weight self-adversarially.

    ``sampler`` defaults to the verbatim copy above; the container driver
    passes TRIX's own ``tasks.negative_sampling`` so the sampling code is
    literally TRIX's (crest precedent). ``support`` is a SupportStore or
    None (reduction mode / phase 1). ``halflink`` is the optional
    ``(mask_answer, mask_query)`` pair of ``model.mask_halflinks``.
    """
    batch = sampler(graph, triples, num_negative, strict=strict)
    pred = model(graph, batch, support=support, walk_offset=walk_offset,
                 halflink=halflink)
    return self_adversarial_nll(pred, num_negative, adversarial_temperature)


def halflink_masks(batch_size: int, p_answer: float, p_query: float,
                   seed: int, step: int, micro: int = 0, device=None):
    """The per-row coins for half-link masking, a pure function of
    (seed, step, micro) so a resumed run draws the same masks. Returns None
    when both probabilities are zero (the loop is then byte-for-byte the old
    one)."""
    if p_answer <= 0.0 and p_query <= 0.0:
        return None
    gen = torch.Generator().manual_seed(int(seed) * 1000003 + int(step) * 7 + int(micro))
    mask_answer = torch.rand(batch_size, generator=gen) < float(p_answer)
    mask_query = torch.rand(batch_size, generator=gen) < float(p_query)
    if device is not None:
        mask_answer, mask_query = mask_answer.to(device), mask_query.to(device)
    return mask_answer, mask_query


def entity_batch_loss(model, graph, batch_size: int, num_negative: int,
                      generator: torch.Generator,
                      adversarial_temperature: float = 1.0, strict: bool = True,
                      support=None, walk_offset: int = 0,
                      sampler=negative_sampling) -> torch.Tensor:
    """One entity step's loss: sample positives, then the objective above."""
    triples = sample_positive_triples(graph, batch_size, generator)
    return entity_loss_from_triples(model, graph, triples, num_negative,
                                    adversarial_temperature, strict, support,
                                    walk_offset, sampler)


def relation_loss_from_triples(model, graph, triples: torch.Tensor,
                               support=None, walk_offset: int = 0) -> torch.Tensor:
    """Softmax cross-entropy over direct relations (design D) on given
    (h, t, r) positives; the softmax penalizes the relation-collapse failure
    MOTIF exhibits on WN-v2. Positives whose relation id is inverse are
    folded to their direct id (the same fact)."""
    num_direct = int(graph.num_relations) // 2
    h, t, r = triples.unbind(-1)
    inv = r >= num_direct
    h2 = torch.where(inv, t, h)
    t2 = torch.where(inv, h, t)
    r2 = torch.where(inv, r - num_direct, r)
    folded = torch.stack([h2, t2, r2], dim=-1)
    scores = model.forward_relation(graph, folded, support=support,
                                    walk_offset=walk_offset)
    return F.cross_entropy(scores, r2)


def relation_batch_loss(model, graph, batch_size: int,
                        generator: torch.Generator, support=None,
                        walk_offset: int = 0) -> torch.Tensor:
    triples = sample_positive_triples(graph, batch_size, generator)  # (h, t, r)
    return relation_loss_from_triples(model, graph, triples, support, walk_offset)
