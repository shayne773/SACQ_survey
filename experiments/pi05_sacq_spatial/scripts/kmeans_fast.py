"""Vectorized drop-in for sacq.vector_quantization.weighted_kmeans_full (same algorithm, same initialisation,
same assignment code, same ridge 1e-4 I, same empty-cluster rule, same clamp): the per-center Python loop
(K masked reductions + K 4x4 solves per iteration) is replaced by index_add_ of the per-sample metrics G_n and of
G_n x_n into (K, 4, 4) / (K, 4) and one batched solve. Mathematically identical; only the floating-point summation
order differs. Used for the large-K bank rungs only (fit.py --fast-kmeans-above)."""
import torch

from sacq.vector_quantization import assign_vectors_full


def weighted_kmeans_full_fast(samples, full_metric, k, iters, seed):
    n, d = samples.shape
    if n < k:
        repeat = (k + n - 1) // max(n, 1)
        return samples.repeat(repeat, 1)[:k].contiguous()
    gen = torch.Generator(device=samples.device)
    gen.manual_seed(seed)
    centers = samples[torch.randperm(n, device=samples.device, generator=gen)[:k]].contiguous()
    eye = torch.eye(d, device=samples.device, dtype=torch.float32)
    gx = torch.einsum("nde,ne->nd", full_metric, samples)
    for _ in range(iters):
        assign = assign_vectors_full(samples, centers, full_metric)
        lhs = torch.zeros(k, d, d, device=samples.device, dtype=torch.float32)
        rhs = torch.zeros(k, d, device=samples.device, dtype=torch.float32)
        lhs.index_add_(0, assign, full_metric)
        rhs.index_add_(0, assign, gx)
        counts = torch.bincount(assign, minlength=k)
        sol, info = torch.linalg.solve_ex(lhs + 1e-4 * eye, rhs)
        bad = (info != 0).nonzero().flatten()
        for j in bad.tolist():
            sol[j] = torch.linalg.lstsq(lhs[j] + 1e-4 * eye, rhs[j][:, None]).solution[:, 0]
        new_centers = torch.where((counts > 0)[:, None], sol, centers)
        centers = new_centers.clamp(-1.0, 1.0).contiguous()
    return centers
