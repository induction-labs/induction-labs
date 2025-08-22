import torch
import torch.nn.functional as F
from flash_attn.cute.interface import flash_attn_varlen_func as fa_cute
from synapse.utils.logging import configure_logging, logging
from torch.autograd import Function

logger = configure_logging(__name__, level=logging.DEBUG)


class CuTeFlashForwardSDPABackward(Function):
    # CuTe Flash Attention forward pass, SDPA backward--varlen interface

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None,
        cu_seqlens_k: torch.Tensor | None,
        seqused_q: torch.Tensor | None = None,
        seqused_k: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int | None, int | None] = (None, None),
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        softcap: float = 0.0,
    ):
        # Use CuTe Flash Attention for forward pass
        output = fa_cute(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            window_size=window_size,
            softcap=softcap,
            causal=causal,
            # dropout_p=0.0,
            # return_attn_probs=False,
        )

        # Save tensors and params for backward pass
        ctx.save_for_backward(q, k, v, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.causal = causal

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Backward pass using SDPA - unpack varlen format first
        q, k, v, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors

        # Detach and require gradients
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)

        with torch.enable_grad():
            batch_size = len(cu_seqlens_q) - 1

            q_unpacked = []
            k_unpacked = []
            v_unpacked = []

            for i in range(batch_size):
                start_q = cu_seqlens_q[i]
                end_q = cu_seqlens_q[i + 1]
                start_k = cu_seqlens_k[i]
                end_k = cu_seqlens_k[i + 1]

                q_seq = q[start_q:end_q]  # [seq_len_q, num_heads, head_dim]
                k_seq = k[start_k:end_k]  # [seq_len_k, num_heads, head_dim]
                v_seq = v[start_k:end_k]  # [seq_len_v, num_heads, head_dim]

                # Pad sequences to max length
                q_padded = F.pad(q_seq, (0, 0, 0, 0, 0, ctx.max_seqlen_q - len(q_seq)))
                k_padded = F.pad(k_seq, (0, 0, 0, 0, 0, ctx.max_seqlen_k - len(k_seq)))
                v_padded = F.pad(v_seq, (0, 0, 0, 0, 0, ctx.max_seqlen_k - len(v_seq)))

                q_unpacked.append(q_padded)
                k_unpacked.append(k_padded)
                v_unpacked.append(v_padded)

            q_batch = torch.stack(
                q_unpacked, dim=0
            )  # [batch, max_seq_q, num_heads, head_dim]
            k_batch = torch.stack(
                k_unpacked, dim=0
            )  # [batch, max_seq_k, num_heads, head_dim]
            v_batch = torch.stack(
                v_unpacked, dim=0
            )  # [batch, max_seq_k, num_heads, head_dim]

            q_sdpa = q_batch.transpose(1, 2)
            k_sdpa = k_batch.transpose(1, 2)
            v_sdpa = v_batch.transpose(1, 2)
            # Print shapes
            # print(
            #     f"q_sdpa: {q_sdpa.shape}, k_sdpa: {k_sdpa.shape}, v_sdpa: {v_sdpa.shape}"
            # )
            # # Print original q, k, v shapes
            # print(
            #     f"q: {q.shape}, k: {k.shape}, v: {v.shape}, cu_seqlens_q: {cu_seqlens_q}, cu_seqlens_k: {cu_seqlens_k}"
            # )

            # Use SDPA for backward pass computation
            # --- ensure Q/K/V have the same # of heads ---
            qh = q_sdpa.size(1)
            kh = k_sdpa.size(1)
            if qh != kh:
                if qh % kh == 0:
                    factor = qh // kh
                    # tile each K/V head group `factor` times
                    k_sdpa = k_sdpa.repeat_interleave(factor, dim=1)
                    v_sdpa = v_sdpa.repeat_interleave(factor, dim=1)
                else:
                    raise RuntimeError(
                        f"Cannot match {qh} query-heads with {kh} key-heads"
                    )

            output_sdpa = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, is_causal=ctx.causal
            )
            output_batch = output_sdpa.transpose(
                1, 2
            )  # [batch, seq_len, num_heads, head_dim]

            output_packed = []
            for i in range(batch_size):
                seq_len = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
                output_seq = output_batch[i, :seq_len]  # Remove padding
                output_packed.append(output_seq)

            output_varlen = torch.cat(output_packed, dim=0)

        grad_tensors = torch.autograd.grad(
            outputs=output_varlen,
            inputs=[q, k, v],
            grad_outputs=grad_outputs,
            retain_graph=False,
        )
        grad_q, grad_k, grad_v = grad_tensors

        return (
            grad_q,
            grad_k,
            grad_v,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# Wrapper function for CuTe forward + SDPA backward varlen attention
def hybrid_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    softcap: float = 0.0,
    **extra_kwargs,
):
    # logger.warning(f"Called with extra kwargs: {extra_kwargs.keys()}")

    return CuTeFlashForwardSDPABackward.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        softmax_scale,
        causal,
        window_size,
        max_seqlen_q,
        max_seqlen_k,
        softcap,
    )
