"""
Tiny backend adapter so the algorithmic cores in seg_crf.py, seg_xlstm.py and
seg_similarity.py are written ONCE and run on either numpy or torch.

Why this exists: the cores (CRF forward algorithm / Viterbi, the mLSTM parallel
scan, the sLSTM stabilised recurrence, similarity + novelty features) are the
parts where a subtle indexing or normalisation error would silently ruin a
multi-hour training run. Writing them against this adapter lets the unit tests
execute the REAL code against brute-force / reference implementations in plain
numpy, with no torch needed. On the training machine the same functions run on
torch tensors (and autograd differentiates straight through them, since every
op here is a differentiable torch op).

Every function takes the array namespace `xp` (NP_OPS or TORCH_OPS) first.
Only the ops the cores actually use are provided.
"""
import numpy as np


class _NumpyOps:
    name = "numpy"

    # reductions / elementwise
    @staticmethod
    def logsumexp(x, axis, keepdims=False):
        m = np.max(x, axis=axis, keepdims=True)
        out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
        return out if keepdims else np.squeeze(out, axis=axis)

    amax = staticmethod(lambda x, axis, keepdims=False: np.max(x, axis=axis, keepdims=keepdims))
    argmax = staticmethod(lambda x, axis: np.argmax(x, axis=axis))
    sum = staticmethod(lambda x, axis, keepdims=False: np.sum(x, axis=axis, keepdims=keepdims))
    cumsum = staticmethod(lambda x, axis: np.cumsum(x, axis=axis))
    exp = staticmethod(np.exp)
    abs = staticmethod(np.abs)
    tanh = staticmethod(np.tanh)
    sqrt = staticmethod(np.sqrt)
    maximum = staticmethod(np.maximum)
    sigmoid = staticmethod(lambda x: 1.0 / (1.0 + np.exp(-x)))
    logsigmoid = staticmethod(lambda x: -np.logaddexp(0.0, -x))
    einsum = staticmethod(np.einsum)
    flip = staticmethod(lambda x, axis: np.flip(x, axis=axis))
    swap_last = staticmethod(lambda x: np.swapaxes(x, -1, -2))
    stack = staticmethod(lambda xs, axis: np.stack(xs, axis=axis))
    concat = staticmethod(lambda xs, axis: np.concatenate(xs, axis=axis))

    # construction helpers (all take a `like` array to inherit dtype/device)
    @staticmethod
    def asarray(a, like):
        return np.asarray(a, dtype=like.dtype)

    @staticmethod
    def zeros(shape, like):
        return np.zeros(shape, dtype=like.dtype)

    @staticmethod
    def full_like(x, value):
        return np.full_like(x, value)

    @staticmethod
    def where(cond_np, a, b):
        """cond_np is a plain numpy bool array (static masks only)."""
        return np.where(cond_np, a, b)

    @staticmethod
    def onehot(labels, k, like):
        return (labels[..., None] == np.arange(k)).astype(like.dtype)

    @staticmethod
    def gather_last(x, idx_np):
        """x: (B, T, N); idx_np: (T, M) static int array -> (B, T, M) with
        out[b, t, m] = x[b, t, idx_np[t, m]]."""
        B = x.shape[0]
        return np.take_along_axis(x, np.broadcast_to(idx_np, (B,) + idx_np.shape), axis=2)

    @staticmethod
    def take_along_row(a, idx):
        """a: (B, K) ints, idx: (B,) ints -> out[b] = a[b, idx[b]]."""
        return a[np.arange(a.shape[0]), idx]

    @staticmethod
    def to_numpy(x):
        return np.asarray(x)


NP_OPS = _NumpyOps()


def get_torch_ops():
    """Built lazily so importing this module never requires torch."""
    import torch
    import torch.nn.functional as F

    class _TorchOps:
        name = "torch"
        logsumexp = staticmethod(lambda x, axis, keepdims=False: torch.logsumexp(x, dim=axis, keepdim=keepdims))
        amax = staticmethod(lambda x, axis, keepdims=False: torch.amax(x, dim=axis, keepdim=keepdims))
        argmax = staticmethod(lambda x, axis: torch.argmax(x, dim=axis))
        sum = staticmethod(lambda x, axis, keepdims=False: torch.sum(x, dim=axis, keepdim=keepdims))
        cumsum = staticmethod(lambda x, axis: torch.cumsum(x, dim=axis))
        exp = staticmethod(torch.exp)
        abs = staticmethod(torch.abs)
        tanh = staticmethod(torch.tanh)
        sqrt = staticmethod(torch.sqrt)
        maximum = staticmethod(torch.maximum)
        sigmoid = staticmethod(torch.sigmoid)
        logsigmoid = staticmethod(F.logsigmoid)
        einsum = staticmethod(torch.einsum)
        flip = staticmethod(lambda x, axis: torch.flip(x, dims=[axis]))
        swap_last = staticmethod(lambda x: x.transpose(-1, -2))
        stack = staticmethod(lambda xs, axis: torch.stack(xs, dim=axis))
        concat = staticmethod(lambda xs, axis: torch.cat(xs, dim=axis))

        @staticmethod
        def asarray(a, like):
            return torch.as_tensor(np.asarray(a), dtype=like.dtype, device=like.device)

        @staticmethod
        def zeros(shape, like):
            return torch.zeros(shape, dtype=like.dtype, device=like.device)

        @staticmethod
        def full_like(x, value):
            return torch.full_like(x, value)

        @staticmethod
        def where(cond_np, a, b):
            cond = torch.as_tensor(np.asarray(cond_np), dtype=torch.bool, device=a.device)
            return torch.where(cond, a, b)

        @staticmethod
        def onehot(labels, k, like):
            return F.one_hot(labels.long(), k).to(like.dtype)

        @staticmethod
        def gather_last(x, idx_np):
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=x.device)
            idx = idx.unsqueeze(0).expand(x.shape[0], -1, -1)
            return torch.gather(x, 2, idx)

        @staticmethod
        def take_along_row(a, idx):
            return a.gather(1, idx.unsqueeze(1)).squeeze(1)

        @staticmethod
        def to_numpy(x):
            return x.detach().cpu().numpy()

    return _TorchOps()