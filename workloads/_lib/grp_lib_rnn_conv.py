"""lib-rnn-conv: recurrent layers (cuDNN / MIOpen RNN), convolution variants (cuDNN / MIOpen conv), pooling,
resampling, normalisation and activation families."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import libkernels as lk

SUITE = lk.Suite("lib-rnn-conv")
op = SUITE.op
D = lk.Data(20250103)

# ---------------------------------------------------------------------------------------------- RNNs
T, B, I, H = 12, 5, 16, 24
XSEQ = D.randn(B, T, I)
LENGTHS = torch.tensor([12, 9, 9, 5, 2])        # sorted descending is not required (enforce_sorted=False below)
H0 = D.randn(2, B, H) * 0.5
C0 = D.randn(2, B, H) * 0.5


def rnn_module(kind, seed, **kw):
    return lk.seeded(seed, lambda: getattr(nn, kind)(I, H, batch_first=True, **kw))


def rnn_outputs(out, state):
    """out, and the final hidden (and cell) state, as one tuple."""
    return (out,) + (tuple(state) if isinstance(state, tuple) else (state,))


@op("lstm_1layer", bound=1e-4)
def _(c):
    m = c.module(rnn_module("LSTM", 1))
    return rnn_outputs(*m(c(XSEQ)))


@op("lstm_2layer_bidirectional_with_initial_state", bound=1e-4)
def _(c):
    m = c.module(rnn_module("LSTM", 2, num_layers=2, bidirectional=True))
    h0 = torch.cat([H0, H0.flip(0)])
    c0 = torch.cat([C0, C0.flip(0)])
    return rnn_outputs(*m(c(XSEQ), (c(h0), c(c0))))


@op("lstm_projection_size", bound=1e-4)
def _(c):
    m = c.module(lk.seeded(3, lambda: nn.LSTM(I, H, proj_size=8, batch_first=True)))
    return rnn_outputs(*m(c(XSEQ)))


@op("gru_1layer", bound=1e-4)
def _(c):
    return rnn_outputs(*c.module(rnn_module("GRU", 4))(c(XSEQ)))


@op("gru_3layer_bidirectional", bound=1e-4)
def _(c):
    return rnn_outputs(*c.module(rnn_module("GRU", 5, num_layers=3, bidirectional=True))(c(XSEQ)))


@op("rnn_tanh_and_relu", bound=1e-4)
def _(c):
    a = c.module(rnn_module("RNN", 6, nonlinearity="tanh"))(c(XSEQ))
    r = c.module(rnn_module("RNN", 7, nonlinearity="relu"))(c(XSEQ))
    return a[0], a[1], r[0], r[1]


@op("lstm_packed_variable_lengths", bound=1e-4)
def _(c):
    m = c.module(rnn_module("LSTM", 8, bidirectional=True))
    packed = pack_padded_sequence(c(XSEQ), LENGTHS, batch_first=True, enforce_sorted=False)
    out, (h, cc) = m(packed)
    padded, lens = pad_packed_sequence(out, batch_first=True, total_length=T)
    return padded, h, cc


@op("gru_packed_variable_lengths", bound=1e-4)
def _(c):
    m = c.module(rnn_module("GRU", 9, num_layers=2))
    packed = pack_padded_sequence(c(XSEQ), LENGTHS, batch_first=True, enforce_sorted=False)
    out, h = m(packed)
    return pad_packed_sequence(out, batch_first=True, total_length=T)[0], h


@op("lstm_backward_grads", bound=1e-3)
def _(c):
    m = c.module(rnn_module("LSTM", 10, bidirectional=True))
    x = c.leaf(XSEQ)
    out, _ = m(x)
    out.pow(2).sum().backward()
    return x.grad, m.weight_hh_l0.grad


@op("lstm_fp16", dtype=torch.float16, bound=2e-2)
def _(c):
    return rnn_outputs(*c.module(rnn_module("LSTM", 11, num_layers=2))(c(XSEQ)))


@op("gru_bf16", dtype=torch.bfloat16, bound=1e-1)
def _(c):
    return rnn_outputs(*c.module(rnn_module("GRU", 12))(c(XSEQ)))


@op("lstm_cell_and_gru_cell", bound=1e-4)
def _(c):
    lc = c.module(lk.seeded(13, lambda: nn.LSTMCell(I, H)))
    gc = c.module(lk.seeded(14, lambda: nn.GRUCell(I, H)))
    h, cell = lc(c(XSEQ[:, 0]))
    return h, cell, gc(c(XSEQ[:, 1]), h)


# ---------------------------------------------------------------------------------------------- convolutions
X2 = D.randn(2, 8, 20, 18)
X1 = D.randn(2, 6, 50)
X3 = D.randn(2, 4, 9, 10, 8)
WDENSE = D.randn(12, 8, 3, 3) * 0.2
WDW = D.randn(8, 1, 3, 3) * 0.3
WGRP = D.randn(12, 4, 3, 3) * 0.2            # groups=2: 8 in / 2 = 4 per group
WT = D.randn(8, 6, 3, 3) * 0.2               # conv_transpose: (in, out/groups, kh, kw)
W3 = D.randn(6, 4, 3, 3, 3) * 0.15
W1 = D.randn(5, 6, 5) * 0.2
W1T = D.randn(6, 5, 3) * 0.2               # conv_transpose1d: (in, out, k)
WT3 = D.randn(4, 3, 3, 3, 3) * 0.15
BIAS = D.randn(12) * 0.1


@op("conv2d_dense_padded_biased", bound=1e-4)
def _(c):
    return F.conv2d(c(X2), c(WDENSE), c(BIAS), padding=1)


@op("conv2d_stride2_asymmetric_kernel", bound=1e-4)
def _(c):
    return F.conv2d(c(X2), c(WDENSE[:, :, :, :2].contiguous()), stride=(2, 3), padding=(1, 0))


@op("conv2d_depthwise", bound=1e-4)
def _(c):
    return F.conv2d(c(X2), c(WDW), padding=1, groups=8)


@op("conv2d_grouped", bound=1e-4)
def _(c):
    return F.conv2d(c(X2), c(WGRP), padding=1, groups=2)


@op("conv2d_dilated_2_and_3", bound=1e-4)
def _(c):
    return F.conv2d(c(X2), c(WDENSE), padding=2, dilation=2), F.conv2d(c(X2), c(WDENSE), padding=3, dilation=3)


@op("conv_transpose2d_stride2_output_padding", bound=1e-4)
def _(c):
    return F.conv_transpose2d(c(X2), c(WT), stride=2, padding=1, output_padding=1)


@op("conv_transpose2d_grouped_dilated", bound=1e-4)
def _(c):
    return F.conv_transpose2d(c(X2), c(WT[:, :3].contiguous()), stride=2, padding=2, dilation=2, groups=1)


@op("conv1d_k5_same", bound=1e-4)
def _(c):
    return F.conv1d(c(X1), c(W1), padding=2)


@op("conv_transpose1d", bound=1e-4)
def _(c):
    return F.conv_transpose1d(c(X1), c(W1T), stride=2)


@op("conv3d_padded", bound=1e-4)
def _(c):
    return F.conv3d(c(X3), c(W3), padding=1)


@op("conv_transpose3d_stride2", bound=1e-4)
def _(c):
    return F.conv_transpose3d(c(X3), c(WT3), stride=2)


@op("conv2d_channels_last", bound=1e-4)
def _(c):
    x = c(X2).contiguous(memory_format=torch.channels_last)
    w = c(WDENSE).contiguous(memory_format=torch.channels_last)
    y = F.conv2d(x, w, padding=1)
    return y.contiguous()          # values are what is compared; the layout is the path under test


@op("conv2d_depthwise_channels_last", bound=1e-4)
def _(c):
    x = c(X2).contiguous(memory_format=torch.channels_last)
    return F.conv2d(x, c(WDW), padding=1, groups=8).contiguous()


@op("conv2d_fp16", dtype=torch.float16, bound=1e-2)
def _(c):
    return F.conv2d(c(X2), c(WDENSE), padding=1)


@op("conv2d_bf16_channels_last", dtype=torch.bfloat16, bound=1e-1)
def _(c):
    return F.conv2d(c(X2).contiguous(memory_format=torch.channels_last), c(WDENSE), padding=1).contiguous()


@op("conv_transpose2d_fp16", dtype=torch.float16, bound=1e-2)
def _(c):
    return F.conv_transpose2d(c(X2), c(WT), stride=2, padding=1)


@op("conv2d_backward_input_and_weight", bound=1e-3)
def _(c):
    x, w = c.leaf(X2), c.leaf(WDENSE)
    F.conv2d(x, w, padding=1).pow(2).sum().backward()
    return x.grad, w.grad


@op("conv_transpose2d_backward", bound=1e-3)
def _(c):
    x, w = c.leaf(X2), c.leaf(WT)
    F.conv_transpose2d(x, w, stride=2, padding=1).pow(2).sum().backward()
    return x.grad, w.grad


@op("conv2d_depthwise_backward", bound=1e-3)
def _(c):
    x, w = c.leaf(X2), c.leaf(WDW)
    F.conv2d(x, w, padding=1, groups=8).pow(2).sum().backward()
    return x.grad, w.grad


@op("conv3d_backward", bound=1e-3)
def _(c):
    x, w = c.leaf(X3), c.leaf(W3)
    F.conv3d(x, w, padding=1).pow(2).sum().backward()
    return x.grad, w.grad


# ---- pooling and resampling
@op("max_pool2d_with_indices", bound=1e-6)
def _(c):
    y, i = F.max_pool2d(c(X2), 3, stride=2, padding=1, return_indices=True)
    return y, i


@op("avg_pool2d_count_include_pad", bound=1e-5)
def _(c):
    return F.avg_pool2d(c(X2), 3, stride=2, padding=1), F.avg_pool2d(c(X2), 3, stride=2, padding=1, count_include_pad=False)


@op("adaptive_avg_and_max_pool2d", bound=1e-5)
def _(c):
    return F.adaptive_avg_pool2d(c(X2), (7, 5)), F.adaptive_max_pool2d(c(X2), (4, 3))


@op("max_pool3d", bound=1e-6)
def _(c):
    return F.max_pool3d(c(X3), 2)


@op("interpolate_bilinear_align_corners_and_not", bound=1e-5)
def _(c):
    return (F.interpolate(c(X2), scale_factor=1.7, mode="bilinear", align_corners=False),
            F.interpolate(c(X2), size=(33, 29), mode="bilinear", align_corners=True))


@op("interpolate_nearest_and_bicubic", bound=1e-4)
def _(c):
    return F.interpolate(c(X2), scale_factor=2, mode="nearest"), F.interpolate(c(X2), size=(25, 23), mode="bicubic", align_corners=False)


@op("unfold_fold_im2col", bound=1e-5)
def _(c):
    cols = F.unfold(c(X2), 3, padding=1, stride=2)
    return cols, F.fold(cols, (20, 18), 3, padding=1, stride=2)


@op("pixel_shuffle_unshuffle", bound=1e-6)
def _(c):
    y = F.pixel_shuffle(c(X2), 2)
    return y, F.pixel_unshuffle(y, 2)


@op("pad_reflect_replicate_circular", bound=1e-6)
def _(c):
    x = c(X2)
    return F.pad(x, (2, 3, 1, 2), mode="reflect"), F.pad(x, (1, 1, 2, 2), mode="replicate"), F.pad(x, (2, 1, 1, 2), mode="circular")


# ---------------------------------------------------------------------------------------------- normalisation
XN = D.randn(6, 8, 14, 12) * 1.5 + 0.5
GAMMA, BETA = D.randn(8) * 0.3 + 1, D.randn(8) * 0.1
RM, RV = D.randn(8) * 0.2, D.rand(8) + 0.5
XT = D.randn(4, 20, 64)
G64, B64 = D.randn(64) * 0.3 + 1, D.randn(64) * 0.1


@op("batch_norm_train_with_running_stats", bound=1e-4)
def _(c):
    rm, rv = c(RM).clone(), c(RV).clone()
    y = F.batch_norm(c(XN), rm, rv, c(GAMMA), c(BETA), training=True, momentum=0.1)
    return y, rm, rv


@op("batch_norm_eval", bound=1e-4)
def _(c):
    return F.batch_norm(c(XN), c(RM), c(RV), c(GAMMA), c(BETA), training=False)


@op("batch_norm_backward", bound=1e-3)
def _(c):
    x, g = c.leaf(XN), c.leaf(GAMMA)
    F.batch_norm(x, None, None, g, c(BETA), training=True).pow(3).sum().backward()
    return x.grad, g.grad


@op("group_norm_forward_backward", bound=1e-3)
def _(c):
    x = c.leaf(XN)
    y = F.group_norm(x, 4, c(GAMMA), c(BETA))
    y.pow(3).sum().backward()
    return y, x.grad


@op("instance_norm", bound=1e-4)
def _(c):
    return F.instance_norm(c(XN), weight=c(GAMMA), bias=c(BETA))


@op("layer_norm_forward_backward", bound=1e-3)
def _(c):
    x = c.leaf(XT)
    y = F.layer_norm(x, (64,), c(G64), c(B64))
    y.pow(3).sum().backward()
    return y, x.grad


@op("rms_norm", bound=1e-4)
def _(c):
    return F.rms_norm(c(XT), (64,), c(G64), eps=1e-6)


@op("local_response_norm", bound=1e-4)
def _(c):
    return F.local_response_norm(c(XN), 3)


@op("layer_norm_bf16", dtype=torch.bfloat16, bound=1e-1)
def _(c):
    return F.layer_norm(c(XT), (64,), c(G64), c(B64))


@op("group_norm_fp16", dtype=torch.float16, bound=1e-2)
def _(c):
    return F.group_norm(c(XN), 4, c(GAMMA), c(BETA))


@op("l2_normalize_cosine_similarity", bound=1e-4)
def _(c):
    return F.normalize(c(XT), dim=-1), F.cosine_similarity(c(XT[:, :10]), c(XT[:, 10:]), dim=-1)


# ---------------------------------------------------------------------------------------------- activations
XA = D.randn(40, 100) * 2.5

for _name, _fn in {
    "relu": F.relu, "leaky_relu": lambda x: F.leaky_relu(x, 0.1), "elu": F.elu, "selu": F.selu, "celu": F.celu,
    "gelu_erf": F.gelu, "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"), "silu": F.silu, "mish": F.mish,
    "softplus": F.softplus, "hardswish": F.hardswish, "hardsigmoid": F.hardsigmoid, "hardtanh": F.hardtanh,
    "sigmoid": torch.sigmoid, "tanh": torch.tanh, "logsigmoid": F.logsigmoid, "softsign": F.softsign,
    "relu6": F.relu6, "tanhshrink": F.tanhshrink, "prelu": lambda x: F.prelu(x, torch.tensor([0.25], dtype=x.dtype, device=x.device)),
}.items():
    @op(f"act_{_name}", bound=1e-4)
    def _(c, _fn=_fn):
        return _fn(c(XA))


@op("act_glu_and_geglu_split", bound=1e-4)
def _(c):
    return F.glu(c(XA)), c(XA)[:, :50] * F.gelu(c(XA)[:, 50:])


@op("softmax_log_softmax_dims", bound=1e-4)
def _(c):
    x = c(XA)
    return torch.softmax(x, 1), torch.log_softmax(x, 0), torch.logsumexp(x, 1)


@op("softmax_backward", bound=1e-3)
def _(c):
    x = c.leaf(XA)
    (torch.softmax(x, 1) * c(XA.flip(1))).sum().backward()
    return x.grad


@op("cross_entropy_label_smoothing", bound=1e-4)
def _(c):
    return F.cross_entropy(c(XA), c.raw(torch.arange(40) % 100), label_smoothing=0.1)


@op("act_gelu_fp16_bf16", dtype=torch.float16, bound=1e-2)
def _(c):
    return F.gelu(c(XA)), F.silu(c(XA))


@op("act_softmax_bf16", dtype=torch.bfloat16, bound=1e-1)
def _(c):
    return torch.softmax(c(XA), 1)


def bench(b):
    dev = lk.DEVICE
    # LSTM: 4 gates x 2 (input + recurrent) GEMMs x 2 flops per MAC, per layer per direction
    bs, t, i, h = b.size(64, 4), b.size(128, 8), b.size(512, 16), b.size(512, 16)
    for tag, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
        m = nn.LSTM(i, h, num_layers=2, bidirectional=True, batch_first=True).to(dev, dt)
        x = torch.randn(bs, t, i, device=dev, dtype=dt)
        flops = bs * t * 2 * (2 * 4 * h * (i + h) + 2 * 4 * h * (2 * h + h))
        with torch.no_grad():
            b.put(f"lstm_2layer_bidir_{tag}_tflops", flops, 1e12, lambda: m(x), iters=5)
    g = nn.GRU(i, h, num_layers=2, batch_first=True).to(dev, torch.float16)
    xg = torch.randn(bs, t, i, device=dev, dtype=torch.float16)
    with torch.no_grad():
        b.put("gru_2layer_fp16_sequences_s", bs, 1, lambda: g(xg), iters=5)
    n, ch, hw = b.size(32, 2), b.size(128, 8), b.size(56, 14)
    x = torch.randn(n, ch, hw, hw, device=dev, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    w = torch.randn(ch, ch, 3, 3, device=dev, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    flops = 2 * n * ch * ch * 9 * hw * hw
    b.put("conv2d_fp16_channels_last_tflops", flops, 1e12, lambda: F.conv2d(x, w, padding=1))
    b.put("conv2d_dilated2_fp16_tflops", flops, 1e12, lambda: F.conv2d(x, w, padding=2, dilation=2))
    wt = torch.randn(ch, ch, 3, 3, device=dev, dtype=torch.float16)
    b.put("conv_transpose2d_s2_fp16_tflops", 2 * n * ch * ch * 9 * hw * hw, 1e12,
          lambda: F.conv_transpose2d(x, wt, stride=2, padding=1, output_padding=1))
    wd = torch.randn(ch, 1, 3, 3, device=dev, dtype=torch.float16)
    b.put("depthwise_conv2d_fp16_gb_s", 2 * x.numel() * 2, 1e9, lambda: F.conv2d(x, wd, padding=1, groups=ch))
    x3 = torch.randn(b.size(8, 1), b.size(64, 4), b.size(32, 8), b.size(56, 8), b.size(56, 8), device=dev, dtype=torch.float16)
    w3 = torch.randn(b.size(64, 4), b.size(64, 4), 3, 3, 3, device=dev, dtype=torch.float16)
    b.put("conv3d_fp16_tflops", 2 * x3.shape[0] * x3.shape[1] * w3.shape[0] * 27 * x3.shape[2] * x3.shape[3] * x3.shape[4], 1e12,
          lambda: F.conv3d(x3, w3, padding=1))
    big = torch.randn(b.size(1 << 26, 1 << 12), device=dev, dtype=torch.float16).view(-1, 1024)
    gam = torch.ones(1024, device=dev, dtype=torch.float16)
    b.put("layer_norm_fp16_gb_s", 2 * big.numel() * 2, 1e9, lambda: F.layer_norm(big, (1024,), gam))
    b.put("softmax_fp16_gb_s", 2 * big.numel() * 2, 1e9, lambda: torch.softmax(big, 1))
    b.put("gelu_fp16_gb_s", 2 * big.numel() * 2, 1e9, lambda: F.gelu(big))
    xb = torch.randn(n, ch, hw, hw, device=dev)
    gm, bt = torch.ones(ch, device=dev), torch.zeros(ch, device=dev)
    b.put("batch_norm_train_fp32_gb_s", 2 * xb.numel() * 4, 1e9, lambda: F.batch_norm(xb, None, None, gm, bt, training=True))
    b.put("group_norm_fp32_gb_s", 2 * xb.numel() * 4, 1e9, lambda: F.group_norm(xb, 8, gm, bt))
