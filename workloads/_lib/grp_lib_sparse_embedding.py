"""lib-sparse-embedding: sparse formats (cuSPARSE / hipSPARSE), embedding bags, scatter / gather / index ops,
sort / top-k / scan / unique (the DLRM-style recommender and sparse-data paths)."""
import torch
import torch.nn.functional as F

import libkernels as lk

SUITE = lk.Suite("lib-sparse-embedding")
op = SUITE.op
D = lk.Data(20250102)

# ---- sparse: ~8% dense matrices with at least one entry per row (a row with none makes softmax ill-defined)
M, K, N = 96, 80, 24
MASK = D.rand(M, K) < 0.08
MASK[torch.arange(M), D.randint(0, K, (M,))] = True
AD = D.randn(M, K) * MASK
BD = D.randn(K, N)
MASK2 = D.rand(K, 40) < 0.1
MASK2[torch.arange(K), D.randint(0, 40, (K,))] = True
A2D = D.randn(K, 40) * MASK2
CD = D.randn(M, N)
XS, YS = D.randn(M, 16), D.randn(16, K)
V = D.randn(M)
DUP_IDX = torch.stack([D.randint(0, 30, (400,)), D.randint(0, 20, (400,))])     # many duplicates
DUP_VAL = D.randn(400)


def masked_softmax_rows(a, mask):
    s = a.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, 1).masked_fill(~mask, 0.0)


@op("spmm_csr_dense", bound=1e-4)
def _(c):
    return c(AD).to_sparse_csr() @ c(BD)


@op("spmm_coo_dense", bound=1e-4)
def _(c):
    return torch.sparse.mm(c(AD).to_sparse(), c(BD))


@op("spmv_csr", bound=1e-4)
def _(c):
    return c(AD).to_sparse_csr() @ c(BD[:, 0])


@op("spmm_csr_transposed_operand", bound=1e-4)
def _(c):
    return torch.mm(c(AD).to_sparse_csr().t().to_sparse_csr(), c(CD))


@op("addmm_csr_beta_alpha", bound=1e-4)
def _(c):
    return torch.addmm(c(CD), c(AD).to_sparse_csr(), c(BD), beta=0.5, alpha=2.0)


@op("spgemm_csr_csr", bound=1e-4)
def _(c):
    return (c(AD).to_sparse_csr() @ c(A2D).to_sparse_csr()).to_dense()


@op("spgemm_coo_coo", bound=1e-4)
def _(c):
    return torch.sparse.mm(c(AD).to_sparse(), c(A2D).to_sparse()).to_dense()


@op("sparse_softmax_coo_dim1", bound=1e-4)
def _(c):
    return torch.sparse.softmax(c(AD).to_sparse(), dim=1).to_dense()


@op("sparse_softmax_reference_dense", bound=1e-4, note="the same masked softmax on a dense tensor, a cross-check")
def _(c):
    return masked_softmax_rows(c(AD), MASK.to(c.dev))


@op("sparse_log_softmax_coo_dim1", bound=1e-4)
def _(c):
    return torch.sparse.log_softmax(c(AD).to_sparse(), dim=1).to_dense()


@op("sparse_sum_dim", bound=1e-4)
def _(c):
    return torch.sparse.sum(c(AD).to_sparse(), dim=1).to_dense()


@op("sparse_coalesce_duplicates", bound=1e-4)
def _(c):
    t = torch.sparse_coo_tensor(c.raw(DUP_IDX), c(DUP_VAL), (30, 20))
    return t.coalesce().to_dense()


@op("sparse_sampled_addmm", bound=1e-4, note="SDDMM; CUDA/ROCm sparse path, not on every backend")
def _(c):
    mask = c(torch.ones(M, K) * MASK).to_sparse_csr()
    return torch.sparse.sampled_addmm(mask, c(XS), c(YS)).to_dense()


@op("sparse_to_dense_roundtrip_csr", bound=1e-6)
def _(c):
    return c(AD).to_sparse_csr().to_dense() + c(AD).to_sparse().to_dense()


# ---- embeddings (DLRM-style): table of 1000 rows x 32, multi-hot bags
ROWS, DIM = 1000, 32
W = D.randn(ROWS, DIM)
IDX = D.randint(0, ROWS, (300,))
LEN = D.randint(1, 12, (24,))
OFF = torch.cat([torch.zeros(1, dtype=torch.long), LEN.cumsum(0)[:-1]])
IDX_BAGS = D.randint(0, ROWS, (int(LEN.sum()),))
PSW = D.rand(int(LEN.sum())) + 0.5
FIXED = D.randint(0, ROWS, (16, 7))
GRAD = D.randn(24, DIM)
PAD_IDX = IDX.clone()
PAD_IDX[::7] = 0


@op("embedding_lookup", bound=1e-6)
def _(c):
    return F.embedding(c.raw(IDX), c(W))


@op("embedding_padding_idx", bound=1e-6)
def _(c):
    return F.embedding(c.raw(PAD_IDX), c(W), padding_idx=0)


for mode in ("sum", "mean", "max"):
    @op(f"embedding_bag_{mode}_offsets", bound=1e-5)
    def _(c, mode=mode):
        return F.embedding_bag(c.raw(IDX_BAGS), c(W), c.raw(OFF), mode=mode)


@op("embedding_bag_sum_per_sample_weights", bound=1e-5)
def _(c):
    return F.embedding_bag(c.raw(IDX_BAGS), c(W), c.raw(OFF), mode="sum", per_sample_weights=c(PSW))


@op("embedding_bag_2d_fixed_bags", bound=1e-5)
def _(c):
    return F.embedding_bag(c.raw(FIXED), c(W), mode="mean")


@op("embedding_bag_backward_dense", bound=1e-4)
def _(c):
    w = c.leaf(W)
    (F.embedding_bag(c.raw(IDX_BAGS), w, c.raw(OFF), mode="sum") * c(GRAD)).sum().backward()
    return w.grad


@op("embedding_backward_repeated_ids", bound=1e-4)
def _(c):
    w = c.leaf(W)
    (F.embedding(c.raw(IDX % 50), w) * c(D_G300)).sum().backward()
    return w.grad


D_G300 = D.randn(300, DIM)

# ---- scatter / gather / index family
SRC = D.randn(64, 20)
INDEX = D.randint(0, 10, (64,))
SCAT = D.rand(64, 20) + 0.5
IDX2D = D.randint(0, 10, (64, 20))
GATH = D.randint(0, 64, (30, 20))
SEL = D.randint(0, 64, (50,))
BIGV = D.randn(5000)


@op("index_add_rows", bound=1e-4)
def _(c):
    return torch.zeros(10, 20, device=c.dev, dtype=c.dtype).index_add_(0, c.raw(INDEX), c(SRC), alpha=0.5)


@op("index_select", bound=1e-6)
def _(c):
    return torch.index_select(c(SRC), 0, c.raw(SEL))


@op("index_put_accumulate", bound=1e-4)
def _(c):
    t = torch.zeros(10, device=c.dev, dtype=c.dtype)
    return t.index_put_((c.raw(INDEX),), c(SRC[:, 0]), accumulate=True)


@op("scatter_add_2d", bound=1e-4)
def _(c):
    return torch.zeros(10, 20, device=c.dev, dtype=c.dtype).scatter_add_(0, c.raw(IDX2D), c(SRC))


for red in ("sum", "prod", "mean", "amax", "amin"):
    @op(f"scatter_reduce_{red}_exclude_self", bound=1e-4)
    def _(c, red=red):
        t = torch.zeros(10, 20, device=c.dev, dtype=c.dtype)
        return t.scatter_reduce(0, c.raw(IDX2D), c(SCAT), reduce=red, include_self=False)


@op("gather_dim0", bound=1e-6)
def _(c):
    return torch.gather(c(SRC), 0, c.raw(GATH))


@op("take_along_dim", bound=1e-6)
def _(c):
    return torch.take_along_dim(c(SRC), c.raw(GATH), dim=0)


@op("masked_select_order", bound=1e-6)
def _(c):
    x = c(SRC)
    return torch.masked_select(x, x > 0.5)


@op("nonzero_ids", kind="int")
def _(c):
    return torch.nonzero(c(SRC) > 1.0)


# ---- sort / top-k / scan / unique
TIES = D.randint(0, 40, (3000,))
TIEF = D.randint(0, 40, (3000,)).float()
PERM_F = D.randn(4096)
MAT = D.randn(48, 200)
CNT = D.randint(0, 100, (5000,))
POS = D.rand(2000) * 3


@op("sort_values_with_ties", bound=1e-6)
def _(c):
    return torch.sort(c(TIEF), stable=True).values


@op("sort_ids_with_ties_stable", kind="int")
def _(c):
    return torch.sort(c(TIEF), stable=True).indices


@op("sort_descending_ids_distinct", kind="int")
def _(c):
    return torch.sort(c(PERM_F), descending=True).indices


@op("sort_rows_values", bound=1e-6)
def _(c):
    return torch.sort(c(MAT), dim=1).values


@op("topk_values", bound=1e-6)
def _(c):
    return torch.topk(c(MAT), 17, dim=1).values


@op("topk_ids", kind="int")
def _(c):
    return torch.topk(c(MAT), 17, dim=1).indices


@op("kthvalue_median", bound=1e-6)
def _(c):
    return torch.kthvalue(c(MAT), 60, dim=1).values, torch.median(c(MAT), dim=1).values


@op("cumsum_rows", bound=1e-4)
def _(c):
    return torch.cumsum(c(MAT), 1)


@op("cumprod_rows", bound=1e-4)
def _(c):
    return torch.cumprod(c(POS[:400].view(20, 20) * 0.7 + 0.4), 1)


@op("cummax_values_and_ids", bound=1e-6)
def _(c):
    r = torch.cummax(c(MAT), 1)
    return r.values, r.indices


@op("cumsum_int64_exact", kind="int")
def _(c):
    return torch.cumsum(c.raw(CNT), 0)


@op("unique_sorted_counts", kind="int")
def _(c):
    u, cnt = torch.unique(c.raw(CNT), return_counts=True)
    return u, cnt


@op("unique_inverse", kind="int")
def _(c):
    return torch.unique(c.raw(CNT), return_inverse=True)[1]


@op("unique_consecutive_counts", kind="int")
def _(c):
    return torch.unique_consecutive(c.raw(CNT // 20), return_counts=True)[1]


@op("bincount_weighted", bound=1e-4)
def _(c):
    return torch.bincount(c.raw(CNT), weights=c(POS.repeat(3)[:5000]), minlength=100)


@op("histc", bound=1e-6)
def _(c):
    return torch.histc(c(POS), bins=30, min=0, max=3)


@op("searchsorted_buckets", kind="int")
def _(c):
    edges = torch.sort(c(BIGV[:256])).values
    return torch.searchsorted(edges, c(BIGV[256:2256]))


@op("argmax_argmin_rows", kind="int")
def _(c):
    return torch.argmax(c(MAT), 1), torch.argmin(c(MAT), 1)


def bench(b):
    dev = lk.DEVICE
    rows, dim = b.size(4_000_000, 2000), 128
    w = torch.randn(rows, dim, device=dev)
    bags, per = b.size(65536, 64), 20
    idx = torch.randint(0, rows, (bags * per,), device=dev)
    off = torch.arange(0, bags * per, per, device=dev)
    gather_bytes = bags * per * dim * 4
    b.put("embedding_bag_sum_gb_s", gather_bytes, 1e9, lambda: F.embedding_bag(idx, w, off, mode="sum"))
    b.put("embedding_lookup_gb_s", gather_bytes, 1e9, lambda: F.embedding(idx, w))
    src = torch.randn(bags * per, dim, device=dev)
    tgt = torch.zeros(rows, dim, device=dev)
    b.put("index_add_gb_s", gather_bytes, 1e9, lambda: tgt.index_add_(0, idx, src))
    b.put("scatter_reduce_amax_gb_s", gather_bytes, 1e9,
          lambda: tgt.scatter_reduce_(0, idx[:, None].expand(-1, dim), src, reduce="amax"))
    n = b.size(1 << 24, 1 << 14)
    x = torch.randn(n, device=dev)
    b.put("sort_mkeys_s", n, 1e6, lambda: torch.sort(x), iters=3)
    b.put("topk_k100_mkeys_s", n, 1e6, lambda: torch.topk(x, 100), iters=3)
    b.put("cumsum_gb_s", 2 * n * 4, 1e9, lambda: torch.cumsum(x, 0))
    xi = torch.randint(0, 1 << 20, (n,), device=dev)
    b.put("unique_mkeys_s", n, 1e6, lambda: torch.unique(xi), iters=2)
    m = b.size(16384, 256)
    dense = torch.randn(m, m, device=dev) * (torch.rand(m, m, device=dev) < 0.01)
    csr, coo, rhs = dense.to_sparse_csr(), dense.to_sparse(), torch.randn(m, 256, device=dev)
    nnz = csr.values().numel()
    b.put("spmm_csr_gflops", 2 * nnz * 256, 1e9, lambda: csr @ rhs)
    b.put("spmm_coo_gflops", 2 * nnz * 256, 1e9, lambda: torch.sparse.mm(coo, rhs))
    b.put("sparse_softmax_coo_gb_s", 3 * nnz * 4, 1e9, lambda: torch.sparse.softmax(coo, dim=1))
