from abc import abstractmethod
from functools import partial

import torch
import torch._dynamo.config
import torch.nn.functional as F


class LigerFusedLinearGRPOBase(torch.autograd.Function):
    @abstractmethod
    def grpo_loss_fn(*args, **kwargs):
        """
        To be extended by subclasses.
        """
        raise NotImplementedError("GRPO loss function must be implemented.")

    @staticmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        tokens,
        tokens_log_prob,
        tokens_mask,
        advantages,
        epsilon=0.2,
        max_seq_len=1024,
        compiled=True,
        chunk_size=1,
    ):
        """Chunked forward pass for GRPO loss computation.

        Args:
            cls: The class
            ctx: Context for backward
            _input: Input tensor [B, T, H]
            weight: Weight tensor [V, H]
            tokens: Token ids tensor [B, T]
            tokens_log_prob: Old policy log probabilities tensor [B, T]
            tokens_mask: Response mask tensor [B, T]
            advantages: Advantages tensor [B]
            epsilon: Clipping parameter for importance sampling ratio
            max_seq_len: Maximum sequence length for normalization
            compiled: Whether to use torch compile
            chunk_size: Size of chunks for processing
        """
        # Initialize accumulators
        loss_acc = torch.zeros((), device=_input.device, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight)  # [V, H]
        grad_inputs = []
        aggregated_metrics = []

        # Create a partial function with fixed arguments
        compute_loss = partial(
            LigerFusedLinearGRPOBase._compute_chunk_loss,
            epsilon=epsilon,
            max_seq_len=max_seq_len,
            grpo_loss_fn=cls.grpo_loss_fn,
        )

        def fused_fwd_bwd(
            input_chunk,
            tokens_chunk,
            tokens_log_prob_chunk,
            tokens_mask_chunk,
        ):
            """Fused forward and backward for a chunk."""
            return torch.func.grad_and_value(compute_loss, argnums=(0, 1), has_aux=True)(
                input_chunk,  # arg 0
                weight,  # arg 1
                tokens_chunk,  # arg 2
                tokens_log_prob_chunk,  # arg 3
                tokens_mask_chunk,  # arg 4
                advantages,  # arg 5
            )

        def accumulate_chunk(
            input_chunk,
            tokens_chunk,
            tokens_log_prob_chunk,
            tokens_mask_chunk,
        ):
            (chunk_grad_input, chunk_grad_weight), (chunk_loss, chunk_metrics) = fused_fwd_bwd(
                input_chunk,
                tokens_chunk,
                tokens_log_prob_chunk,
                tokens_mask_chunk,
            )

            # this part makes no difference for batch dim chunking or sequence dim chunking
            # Accumulate gradients and loss
            grad_weight.add_(chunk_grad_weight)
            grad_inputs.append(chunk_grad_input)
            loss_acc.add_(chunk_loss)
            # Initialize storage for metrics on first chunk
            if len(aggregated_metrics) == 0:
                for metric in chunk_metrics:
                    if metric.ndim == 0:
                        aggregated_metrics.append(torch.zeros((), device=metric.device))
                    else:
                        aggregated_metrics.append([])

            # Accumulate metrics
            for i, metric in enumerate(chunk_metrics):
                if metric.ndim == 0:
                    aggregated_metrics[i].add_(metric)
                else:
                    aggregated_metrics[i].append(metric)

        if compiled:
            fused_fwd_bwd = torch.compile(fused_fwd_bwd)

        # Process input in chunks based on chunk_size
        chunks = max(1, _input.shape[1] // chunk_size)
        _input_chunks = torch.chunk(_input, chunks=chunks, dim=1)
        _tokens_chunks = torch.chunk(tokens, chunks=chunks, dim=1)
        _tokens_log_prob_chunks = torch.chunk(tokens_log_prob, chunks=chunks, dim=1)
        _tokens_mask_chunks = torch.chunk(tokens_mask, chunks=chunks, dim=1)
           

        for (
            input_chunk,
            tokens_chunk,
            tokens_log_prob_chunk,
            tokens_mask_chunk,
        ) in zip(
            _input_chunks,
            _tokens_chunks,
            _tokens_log_prob_chunks,
            _tokens_mask_chunks,
        ):
            # Mark dynamic dimensions
            torch._dynamo.mark_dynamic(input_chunk, 1)
            torch._dynamo.mark_dynamic(tokens_chunk, 1)
            torch._dynamo.mark_dynamic(tokens_log_prob_chunk, 1)
            torch._dynamo.mark_dynamic(tokens_mask_chunk, 1)
            
            accumulate_chunk(
                input_chunk,
                tokens_chunk,
                tokens_log_prob_chunk,
                tokens_mask_chunk,
            )

        # Combine gradients
        grad_input = torch.cat(grad_inputs, dim=1)

        # Save for backward
        ctx.save_for_backward(grad_input, grad_weight)

        # Finalize metrics
        final_metrics = []
        for metric in aggregated_metrics:
            if isinstance(metric, list):
                final_metrics.append(torch.cat(metric, dim=1))
            else:
                final_metrics.append(metric)

        return loss_acc, tuple(final_metrics)

    @staticmethod
    def _compute_chunk_loss(
        input_chunk,
        weight,
        tokens_chunk,
        tokens_log_prob_chunk,
        tokens_mask_chunk,
        advantages,
        epsilon=0.2,
        max_seq_len=1024,
        grpo_loss_fn=None,
    ):
        """Compute loss for a single chunk."""
        # Get policy log probabilities using chunk_forward
        log_probs, _ = LigerFusedLinearGRPOBase.chunk_forward(input_chunk, weight)

        # Compute chunk loss and metrics using the provided loss function  
        chunk_loss, chunk_metrics = grpo_loss_fn(
            log_probs=log_probs,
            tokens=tokens_chunk,
            tokens_log_prob=tokens_log_prob_chunk,
            tokens_mask=tokens_mask_chunk,
            advantages=advantages, # no chunks for advantages yet since we plan to chunk in sequence dim
            epsilon=epsilon,
            max_seq_len=max_seq_len,
        )

        return chunk_loss, chunk_metrics

    @staticmethod
    def chunk_forward(input_chunk, weight):
        """Forward pass computation for a single chunk without explicit reshaping."""
        # Directly compute logits via batched matrix multiplication: [B, T, H] @ [H, V] -> [B, T, V]
        logits = torch.matmul(input_chunk, weight.t())

        # Compute log probabilities using softmax over the last dimension
        log_probs = F.log_softmax(logits.float(), dim=-1)

        return log_probs, logits

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for GRPO loss."""
        grad_input, grad_weight = ctx.saved_tensors

        if grad_output != 1.0:
            grad_input = grad_input * grad_output
            grad_weight = grad_weight * grad_output

        return (
            grad_input,
            grad_weight,
            None,  # grad_tokens
            None,  # grad_tokens_log_prob
            None,  # grad_tokens_mask
            None,  # grad_advantages
            None,  # grad_epsilon
            None,  # grad_max_seq_len
            None,  # grad_compiled
            None,  # grad_chunk_size
        )


class LigerFusedLinearGRPOFunction(LigerFusedLinearGRPOBase):
    @staticmethod
    def grpo_loss_fn(
        log_probs,
        tokens,
        tokens_log_prob,
        tokens_mask,
        advantages,
        epsilon=0.2,
        max_seq_len=1024,
        **kwargs,
    ):
        """
        Original GRPO loss: clipped importance-sampling objective without KL penalty,
        normalizing sum of per-token losses by a fixed max sequence length.
        """
        # Get current policy log probs for selected tokens
        current_log_probs = log_probs.gather(
            dim=-1,
            index=tokens.unsqueeze(-1),
        ).squeeze(-1)  # (batch_size, seq_len)

        # importance-sampling ratio and clipped ratio
        prob_ratio = torch.exp(current_log_probs - tokens_log_prob)
        clipped = torch.clamp(prob_ratio, 1.0 - epsilon, 1.0 + epsilon)

        # per-token loss: negative clipped objective
        per_token_loss = -torch.min(
            prob_ratio * advantages.unsqueeze(1),
            clipped * advantages.unsqueeze(1),
        )
        # mask padding positions
        per_token_loss = per_token_loss * tokens_mask
        # Return sum without normalization - normalization handled in forward
        loss = per_token_loss.sum()
        return loss, []
    
    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        tokens,
        tokens_log_prob,
        tokens_mask,
        advantages,
        epsilon=0.2,
        max_seq_len=1024,
        compiled=True,
        chunk_size=1,
    ):
        """Fused forward pass for GRPO loss computation.

        Args:
            ctx: Context for backward
            _input: Input tensor [B, T, H]
            weight: Weight tensor [V, H]
            tokens: Token ids tensor [B, T]
            tokens_log_prob: Old policy log probabilities tensor [B, T]
            tokens_mask: Response mask tensor [B, T]
            advantages: Advantages tensor [B]
            epsilon: Clipping parameter for importance ratio
            max_seq_len: Maximum sequence length for normalization
            compiled: Whether to use torch compile
            chunk_size: Size of chunks for memory-efficient processing

        Returns:
            loss: Computed loss
            metrics: Computed metrics
        """
        return super().forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            tokens=tokens,
            tokens_log_prob=tokens_log_prob,
            tokens_mask=tokens_mask,
            advantages=advantages,
            epsilon=epsilon,
            max_seq_len=max_seq_len,
            compiled=compiled,
            chunk_size=chunk_size,
        )


class LigerFusedLinearGRPOLoss(torch.nn.Module):
    """
    Fused linear layer with GRPO (Group Relative Policy Optimization) loss.
    
    This module combines the linear transformation and GRPO loss computation
    in a memory-efficient manner using chunked processing.
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        max_seq_len: int = 1024,
        compiled: bool = True,
        chunk_size: int = 1,
    ):
        """
        Initialize the GRPO loss module.
        
        Args:
            epsilon: Clipping parameter for importance ratio
            max_seq_len: Maximum sequence length for normalization
            compiled: Whether to use torch compile for optimization
            chunk_size: Size of chunks for memory-efficient processing
        """
        super().__init__()
        self.epsilon = epsilon
        self.max_seq_len = max_seq_len
        self.compiled = compiled
        self.chunk_size = chunk_size

    def forward(
        self,
        _input,
        lin_weight,
        tokens,
        tokens_log_prob,
        tokens_mask,
        advantages,
    ):
        """
        Forward pass for GRPO loss computation.

        Notes:
            - tokens, tokens_log_prob, tokens_mask should already be aligned
         
        Args:
            _input: Input embeddings [B, T, H]
            lin_weight: Linear layer weight [V, H]
            tokens: Token IDs for responses [B, T]
            tokens_log_prob: Old policy per-token log probs [B, T]
            tokens_mask: Response mask [B, T]
            advantages: Advantage values [B]
        """
        return LigerFusedLinearGRPOFunction.apply(
            _input,
            lin_weight,
            tokens,
            tokens_log_prob,
            tokens_mask,
            advantages,
            self.epsilon,
            self.max_seq_len,
            self.compiled,
            self.chunk_size,
        )