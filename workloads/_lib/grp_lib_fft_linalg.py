"""lib-fft-linalg: torch.fft (cuFFT / rocFFT) and torch.linalg (cuSOLVER, cuBLAS / rocSOLVER, rocBLAS)."""
import math

import torch
import torch.nn.functional as F

import libkernels as lk

SUITE = lk.Suite("lib-fft-linalg")
op = SUITE.op
D = lk.Data(20250101)


def cplx(c, re, im):
    return torch.complex(c(re), c(im))


# ---- FFT: sizes chosen to hit different plans: power of two, mixed radix (odd), prime (Bluestein / chirp-z)
R256, I256 = D.randn(3, 256), D.randn(3, 256)
R225, I225 = D.randn(3, 225), D.randn(3, 225)          # 3^2 * 5^2
R127, I127 = D.randn(2, 127), D.randn(2, 127)          # prime
R250 = D.randn(4, 250)                                  # even, not a power of two
R251 = D.randn(4, 251)                                  # odd
R2D, I2D = D.randn(2, 48, 40), D.randn(2, 48, 40)
R2R = D.randn(2, 33, 50)                                # odd x even
R3D, I3D = D.randn(8, 12, 10), D.randn(8, 12, 10)
RCOL = D.randn(64, 7)
CA, CB = D.randn(2, 300), D.randn(2, 41)
RH = D.randn(3, 64)


@op("fft_c2c_pow2", bound=2e-5)
def _(c):
    return torch.fft.fft(cplx(c, R256, I256))


@op("fft_c2c_odd_225", bound=2e-5)
def _(c):
    return torch.fft.fft(cplx(c, R225, I225))


@op("fft_c2c_prime_127", bound=2e-5)
def _(c):
    return torch.fft.fft(cplx(c, R127, I127))


@op("ifft_c2c_odd_225", bound=2e-5)
def _(c):
    return torch.fft.ifft(cplx(c, R225, I225))


@op("fft_real_input_pow2", bound=2e-5)
def _(c):
    return torch.fft.fft(c(R256))


@op("rfft_even_250", bound=2e-5)
def _(c):
    return torch.fft.rfft(c(R250))


@op("rfft_odd_251", bound=2e-5)
def _(c):
    return torch.fft.rfft(c(R251))


@op("irfft_odd_roundtrip_251", bound=2e-5)
def _(c):
    # an odd output length must be asked for explicitly: n=251 from the 126 bins of rfft
    return torch.fft.irfft(torch.fft.rfft(c(R251)), n=251)


@op("fft_ortho_norm_dim0", bound=2e-5)
def _(c):
    return torch.fft.fft(cplx(c, RCOL, RCOL.flip(0)), dim=0, norm="ortho")


@op("fft2_c2c", bound=2e-5)
def _(c):
    return torch.fft.fft2(cplx(c, R2D, I2D))


@op("rfft2_odd_by_even", bound=2e-5)
def _(c):
    return torch.fft.rfft2(c(R2R))


@op("irfft2_roundtrip", bound=2e-5)
def _(c):
    return torch.fft.irfft2(torch.fft.rfft2(c(R2R)), s=(33, 50))


@op("fftn_3d_c2c", bound=2e-5)
def _(c):
    return torch.fft.fftn(cplx(c, R3D, I3D))


@op("fft_shift_and_freq", bound=1e-5)
def _(c):
    return torch.fft.fftshift(torch.fft.fft(c(R256[:, :64]))) * torch.fft.fftfreq(64, 0.5, dtype=c.dtype, device=c.dev)


@op("hfft_hermitian_input", bound=2e-5)
def _(c):
    return torch.fft.hfft(cplx(c, RH, RH.flip(1)), n=126)


@op("fft_convolution_vs_direct", bound=5e-5)
def _(c):
    # linear convolution through rfft*rfft->irfft, checked against F.conv1d in the same dtype
    a, k = c(CA), c(CB)
    n = a.shape[-1] + k.shape[-1] - 1
    via_fft = torch.fft.irfft(torch.fft.rfft(a, n) * torch.fft.rfft(k, n), n)
    direct = F.conv1d(F.pad(a, (k.shape[-1] - 1,) * 2).unsqueeze(0), k.flip(-1).unsqueeze(1), groups=2)[0]
    return via_fft, direct


@op("fft_half_pow2", dtype=torch.float16, bound=2e-2, note="cuFFT/rocFFT half precision supports power-of-two sizes only")
def _(c):
    return torch.fft.fft(c(R256[:, :128]))


# ---- dense linear algebra. Matrices are made well conditioned (an identity multiple added), so the
# float32 error is a small multiple of 1e-7 and a wrong result is not hidden by conditioning.
def spd(n):
    m = D.randn(n, n)
    return m @ m.T / n + torch.eye(n)


S64, S33 = spd(64), spd(33)
G48 = D.randn(48, 48) + 4 * torch.eye(48)
B48 = D.randn(48, 5)
T40 = torch.tril(D.randn(40, 40)) + 3 * torch.eye(40)
B40 = D.randn(40, 6)
TALL = D.randn(80, 24)
BT = D.randn(80, 3)
SQ32 = D.randn(32, 32)
BATCH = D.randn(6, 20, 20) + 3 * torch.eye(20)
SMALL = D.randn(24, 24) * 0.15
GA, GB = D.randn(37, 53), D.randn(53, 29)


@op("cholesky", bound=1e-4)
def _(c):
    return torch.linalg.cholesky(c(S64))


@op("cholesky_solve", bound=1e-4)
def _(c):
    return torch.cholesky_solve(c(B48[:33]), torch.linalg.cholesky(c(S33)))


@op("cholesky_batched_inverse", bound=1e-4)
def _(c):
    s = c(BATCH @ BATCH.transpose(-1, -2) / 20 + torch.eye(20))
    return torch.cholesky_inverse(torch.linalg.cholesky(s))


@op("qr_reduced", bound=1e-4)
def _(c):
    q, r = torch.linalg.qr(c(TALL))
    # the signs of Q's columns / R's rows are a convention: compare |R|, and Q R (which must give back TALL)
    return r.abs(), q @ r


@op("qr_complete_orthogonality", bound=1e-4)
def _(c):
    q, _ = torch.linalg.qr(c(TALL), mode="complete")
    return q.T @ q                                     # must be the identity


@op("svd_values_and_reconstruction", bound=1e-4)
def _(c):
    u, s, vh = torch.linalg.svd(c(TALL), full_matrices=False)
    return s, (u * s) @ vh


@op("svdvals_square", bound=1e-4)
def _(c):
    return torch.linalg.svdvals(c(SQ32))


# bound 3e-4, was 1e-4: the first real GPU (A10G, cuSOLVER syevd, torch 2.10.0+cu130) measured 1.44e-4 of rms against the float64
# CPU result. 3e-4 is about 2x that, still ~100x tighter than a wrong eigendecomposition; the 1e-4 was derived from CPU runs only.
@op("eigh_values_and_reconstruction", bound=3e-4)
def _(c):
    w, v = torch.linalg.eigh(c(S64))
    return w, (v * w) @ v.T


@op("eigvalsh_lower", bound=1e-4)
def _(c):
    return torch.linalg.eigvalsh(c(S33), UPLO="L")


@op("eigvals_general_sorted_magnitude", bound=1e-3)
def _(c):
    # general (non-symmetric) eigenvalues come in LAPACK/MAGMA order: compare the sorted moduli
    return torch.linalg.eigvals(c(SQ32)).abs().sort().values


@op("lu_factor_lu_solve", bound=2e-4)
def _(c):
    lu, piv = torch.linalg.lu_factor(c(G48))
    return torch.linalg.lu_solve(lu, piv, c(B48))


@op("lu_reconstruction", bound=1e-4)
def _(c):
    p, l, u = torch.linalg.lu(c(G48))
    return p @ l @ u


@op("solve_batched", bound=2e-4)
def _(c):
    return torch.linalg.solve(c(BATCH), c(BATCH[:, :, :3]))


@op("inv", bound=2e-4)
def _(c):
    return torch.linalg.inv(c(G48))


@op("det_slogdet", bound=1e-4)
def _(c):
    s, ld = torch.linalg.slogdet(c(G48))
    return s, ld


@op("solve_triangular", bound=1e-4)
def _(c):
    return torch.linalg.solve_triangular(c(T40), c(B40), upper=False)


@op("lstsq_tall", bound=1e-3)
def _(c):
    return torch.linalg.lstsq(c(TALL), c(BT)).solution


@op("pinv_tall", bound=1e-3)
def _(c):
    return torch.linalg.pinv(c(TALL))


@op("matrix_exp", bound=1e-4)
def _(c):
    return torch.linalg.matrix_exp(c(SMALL))


@op("matrix_power_and_norms", bound=1e-4)
def _(c):
    m = c(SMALL * 3)
    return torch.linalg.matrix_power(m, 5), torch.linalg.matrix_norm(m), torch.linalg.matrix_norm(m, "nuc"), torch.linalg.vector_norm(m, 3)


@op("matmul_gemm_odd_shapes", bound=1e-4)
def _(c):
    return c(GA) @ c(GB)


def bench(b):
    dev = lk.DEVICE
    # FFT: GFLOP/s by the usual 5 N log2 N convention (complex-to-complex), GB/s of data moved
    n = b.size(1 << 24, 1 << 12)
    x = torch.randn(n, device=dev, dtype=torch.complex64)
    b.put("fft_c2c_1d_pow2_gflops", 5 * n * (n.bit_length() - 1), 1e9, lambda: torch.fft.fft(x))
    xr = torch.randn(n, device=dev)
    b.put("rfft_1d_pow2_gflops", 2.5 * n * (n.bit_length() - 1), 1e9, lambda: torch.fft.rfft(xr))
    m = 1000003 if not b.tiny else 997                    # a prime: Bluestein path
    xp = torch.randn(m, device=dev, dtype=torch.complex64)
    b.put("fft_c2c_1d_prime_gflops", 5 * m * math.log2(m), 1e9, lambda: torch.fft.fft(xp))
    s = b.size(4096, 64)
    x2 = torch.randn(s, s, device=dev, dtype=torch.complex64)
    b.put("fft2_c2c_gflops", 5 * s * s * math.log2(s * s), 1e9, lambda: torch.fft.fft2(x2))
    # dense linear algebra: flops by the LAPACK counts; matrices per second for the batched small cases
    k = b.size(4096, 96)
    a = torch.randn(k, k, device=dev)
    spd_ = a @ a.T / k + torch.eye(k, device=dev)
    b.put("cholesky_n4096_gflops", k ** 3 / 3, 1e9, lambda: torch.linalg.cholesky(spd_), iters=3)
    b.put("qr_n4096_gflops", 4 / 3 * k ** 3, 1e9, lambda: torch.linalg.qr(a), iters=2)
    b.put("lu_factor_n4096_gflops", 2 / 3 * k ** 3, 1e9, lambda: torch.linalg.lu_factor(a), iters=3)
    b.put("eigh_n4096_ops_s", 1, 1, lambda: torch.linalg.eigh(spd_), iters=1)
    b.put("svd_n4096_ops_s", 1, 1, lambda: torch.linalg.svd(a), iters=1)
    nb = b.size(4096, 16)
    batch = torch.randn(nb, 32, 32, device=dev) + 4 * torch.eye(32, device=dev)
    b.put("lu_solve_batched_32x32_systems_s", nb, 1, lambda: torch.linalg.solve(batch, batch[:, :, :1]))
    b.put("matrix_exp_batched_32x32_ops_s", nb, 1, lambda: torch.linalg.matrix_exp(batch * 0.05))

