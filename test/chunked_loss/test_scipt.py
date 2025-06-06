import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_grpo import LigerFusedLinearGRPOLoss, LigerFusedLinearGRPOFunction, LigerFusedLinearGRPOFunction
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose
from test.utils import set_seed
liger_fused_linear_grpo = LigerFusedLinearGRPOFunction.apply

device = infer_device()

# set random seed globally
set_seed()


class TorchLMHeadGRPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        beta: float = 0.1,
        epsilon: float = 0.2,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.beta = beta
        self.epsilon = epsilon
        self.max_seq_len = max_seq_len

    def forward(
        self,
        x,  # Shape: [batch_size, seq_len, hidden_size]
        tokens,  # Shape: [batch_size, seq_len]
        tokens_log_prob,  # Shape: [batch_size, seq_len]
        tokens_mask,  # Shape: [batch_size, seq_len]
        advantages,  # Shape: [batch_size,]
    ):
        logits = x @ self.lin.weight.t()
        if self.lin.bias is not None:
            logits = logits + self.lin.bias.float()
        # Get log probabilities
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # Get chosen token probabilities
        per_token_logps = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)


        # Compute policy gradient loss with importance sampling ratio
        coef_1 = torch.exp(per_token_logps - tokens_log_prob)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        loss = (per_token_loss * tokens_mask).sum() / torch.clamp(tokens_mask.sum(), min=1.0)

        return loss, []


class LigerLMHeadGRPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        epsilon: float = 0.2,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.grpo_loss = LigerFusedLinearGRPOLoss(
            epsilon=epsilon,
            max_seq_len=max_seq_len,
            compiled=True,
            chunk_size=1,
        )

    def forward(
        self,
        x,
        tokens,
        tokens_log_prob,
        tokens_mask,
        advantages,
        
    ):
        # Pass only the arguments defined in LigerFusedLinearGRPOFunction.forward()
        return self.grpo_loss(
            x,  # _input
            self.lin.weight,  # weight
            tokens,  # tokens
            tokens_log_prob,  # tokens_log_prob
            tokens_mask,  # tokens_mask
            advantages,  # advantages
            self.lin.bias,  # bias
        )

def test():
    """Single test case for GRPO loss correctness with fixed parameters."""
    # Fixed test parameters
    B, T, H, V = 8, 128, 1024, 4096 # batch, seq_len, hidden_size, vocab_size
    scalar = 1.0
    dtype = torch.float32
    atol, rtol = 1e-5, 5e-4
    bias = False
    epsilon = 0.2
    max_seq_len = 1024
    
    # Reset torch compiler cache for each parameter of the test case
    torch.compiler.reset()

    torch_lm_head_grpo = TorchLMHeadGRPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        epsilon=epsilon,
        max_seq_len=max_seq_len,
    )
    liger_lm_head_grpo = LigerLMHeadGRPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        epsilon=epsilon,
        max_seq_len=max_seq_len,
    )

    # Initialize weights
    torch_lm_head_grpo.lin.weight.data = liger_lm_head_grpo.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )
    if bias:
        torch_lm_head_grpo.lin.bias.data = liger_lm_head_grpo.lin.bias.data = torch.randn(V, device=device, dtype=dtype)

    # Create inputs with shape [B, T, H]
    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    # Create selected token ids with shape [B, T]
    tokens = torch.randint(0, V, (B, T), device=device)

    # Compute per-token logps
    with torch.no_grad():
        logits = _input @ torch_lm_head_grpo.lin.weight.t()
        if torch_lm_head_grpo.lin.bias is not None:
            logits = logits + torch_lm_head_grpo.lin.bias
        logps = F.log_softmax(logits.float(), dim=-1)
        per_token_logps = logps.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)

    # Create attention mask with random padding [B, T]
    attention_mask = torch.ones(B, T, device=device)
    num_elements_to_mask = torch.randint(1, B * T // 2, (1,)).item()
    mask_indices = torch.randperm(B * T)[:num_elements_to_mask]
    attention_mask.view(-1)[mask_indices] = 0

    # Create advantages with shape [B]
    advantages = torch.rand(B, device=device, dtype=dtype)

    # Forward pass with reference model
    loss1, aux1 = torch_lm_head_grpo(
        input1,
        tokens,
        per_token_logps,
        attention_mask,
        advantages,
    )
    loss2, aux2 = liger_lm_head_grpo(
        input2,
        tokens,
        per_token_logps,
        attention_mask,
        advantages,
    )
    # Check losses match
    assert not torch.isnan(loss1)
    assert not torch.isnan(loss2)
    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    # Check metrics match
    assert len(aux1) == len(aux2)
    # aggregated metrics are unstable for bfloat16
    for metric1, metric2 in zip(aux1, aux2):
        assert_verbose_allclose(metric1, metric2, atol=atol, rtol=rtol)

    # Backward pass
    loss1.backward()
    loss2.backward()

    # Check gradients match for loss_type
    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head_grpo.lin.weight.grad,
        liger_lm_head_grpo.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    if bias:
        assert_verbose_allclose(
            torch_lm_head_grpo.lin.bias.grad,
            liger_lm_head_grpo.lin.bias.grad,
            atol=atol,
            rtol=rtol,
        )

if __name__ == "__main__":
    test()