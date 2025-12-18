import torch
from typing import List, Sequence, Tuple, Union


TensorOrSequence = Union[torch.Tensor, Sequence[torch.Tensor]]


def _clone_with_noise(x: TensorOrSequence, noise_level: float) -> TensorOrSequence:
    """Clone x and optionally add Gaussian noise."""
    if x is None:
        return None
    if isinstance(x, (tuple, list)):
        return type(x)(_clone_with_noise(elem, noise_level) for elem in x)
    x_clone = x.clone()
    if noise_level:
        x_clone = x_clone + noise_level * torch.randn_like(x_clone)
    return x_clone


def _infer_batch_size(x: TensorOrSequence, fallback: TensorOrSequence = None) -> int:
    """Infer batch size from the leading dimension of a tensor or tensor sequence."""
    candidate = x if x is not None else fallback
    if candidate is None:
        raise ValueError("Unable to infer batch size without x_in or x_init.")
    if isinstance(candidate, (tuple, list)) and candidate:
        candidate = candidate[0]
    if torch.is_tensor(candidate):
        return int(candidate.shape[0]) if candidate.shape else 1
    raise ValueError("x_in or x_init must be a torch.Tensor (or sequence of tensors).")


def _wrap_reverse_call(
    model,
    B: int,
    x_in: TensorOrSequence,
    cond,
    intermediate_every: int,
    mask_override,
) -> Tuple:
    """Call reverse_samples for conditional models, supporting tuple inputs."""
    if isinstance(x_in, (tuple, list)):
        return model.reverse_samples(
            B,
            *x_in,
            cond,
            intermediate_every=intermediate_every,
            mask_override=mask_override,
        )
    return model.reverse_samples(
        B,
        x_in,
        cond,
        intermediate_every=intermediate_every,
        mask_override=mask_override,
    )


def _get_intermediate_indices(result: Sequence) -> List[int]:
    """
    Identify which outputs correspond to intermediate trajectories.
    Handles standard conditional samplers (index 1) and mixed samplers (indices 2, 3).
    """
    if len(result) == 4 and all(isinstance(result[i], (list, tuple)) for i in (2, 3)):
        return [2, 3]
    if len(result) >= 2 and isinstance(result[1], (list, tuple)):
        return [1]
    return []


def _select_init_component(x_init: TensorOrSequence, component_idx: int):
    """Pick the relevant slice of x_init for multi-trajectory outputs."""
    if x_init is None:
        return None
    if isinstance(x_init, (tuple, list)):
        if component_idx < len(x_init):
            return x_init[component_idx]
        return None
    return x_init


def _maybe_truncate_intermediates(
    intermediates: List[torch.Tensor], num_steps: int
) -> List[torch.Tensor]:
    """Return the last num_steps elements, if requested."""
    if not num_steps or num_steps <= 0:
        return intermediates
    if len(intermediates) <= num_steps:
        return intermediates
    return intermediates[-num_steps:]


def _process_sampling_result(
    raw_result: Tuple,
    x_init_noisy: TensorOrSequence,
    num_steps: int,
) -> Tuple:
    """
    Adjust the returned intermediates:
    - seed the first recorded state with x_init_noisy when provided
    - truncate to the last num_steps (reusing the model sampler and early-breaking via slicing)
    - update final outputs to match the truncated chains
    """
    if not isinstance(raw_result, (list, tuple)):
        return raw_result
    result = list(raw_result)
    intermediate_indices = _get_intermediate_indices(result)
    if not intermediate_indices:
        return raw_result

    for idx, seq_idx in enumerate(intermediate_indices):
        seq = list(result[seq_idx])
        init_component = _select_init_component(x_init_noisy, idx)
        if init_component is not None and seq:
            seq[0] = init_component
        seq = _maybe_truncate_intermediates(seq, num_steps)
        # Preserve container type (list vs tuple) for callers expecting that.
        result[seq_idx] = type(result[seq_idx])(seq) if isinstance(result[seq_idx], tuple) else seq

    if num_steps and intermediate_indices == [1] and result[1]:
        result[0] = result[1][-1]
    if num_steps and intermediate_indices == [2, 3]:
        if result[2]:
            result[0] = result[2][-1]
        if result[3]:
            result[1] = result[3][-1]

    return tuple(result) if isinstance(raw_result, tuple) else result


def run_sampling(
    model,
    x_in: TensorOrSequence,
    cond,
    mode: str = "cond",
    num_steps: int = None,
    x_init: TensorOrSequence = None,
    noise_level: float = 0.0,
    intermediate_every: int = 0,
    mask_override=None,
):
    """
    Helper wrapper for model.reverse_samples with optional warm-starts and truncation.

    Args:
        mode: "cond" for discrete/conditional samplers, "cont" for continuous samplers.
        num_steps: when set, returns only the last num_steps intermediates by reusing the
            existing sampler and truncating the saved chain (equivalent to early-break).
        x_init: optional initial state to inject (recorded as the first intermediate after
            adding Gaussian noise scaled by noise_level).
    """
    x_init_noisy = _clone_with_noise(x_init, noise_level)
    B = _infer_batch_size(x_in, x_init_noisy)

    if mode == "cond":
        # If no extras requested, just delegate directly to keep outputs identical.
        if num_steps is None and x_init_noisy is None:
            return _wrap_reverse_call(
                model,
                B,
                x_in,
                cond,
                intermediate_every=intermediate_every,
                mask_override=mask_override,
            )

        # Ensure we have enough stored steps to truncate; fall back to dense saves if needed.
        call_intermediate_every = (intermediate_every or 1) if num_steps else intermediate_every
        result = _wrap_reverse_call(
            model,
            B,
            x_in,
            cond,
            intermediate_every=call_intermediate_every,
            mask_override=mask_override,
        )
        if num_steps:
            intermediate_indices = _get_intermediate_indices(result if isinstance(result, (list, tuple)) else [])
            min_len = min((len(result[idx]) for idx in intermediate_indices), default=0)
            if min_len < num_steps and call_intermediate_every != 1:
                result = _wrap_reverse_call(
                    model,
                    B,
                    x_in,
                    cond,
                    intermediate_every=1,
                    mask_override=mask_override,
                )

        processed = _process_sampling_result(result, x_init_noisy, num_steps or 0)
        return processed

    if mode == "cont":
        target_steps = num_steps if num_steps is not None else getattr(model, "max_diffusion_steps", None)
        # When x_init is supplied, explicitly disable masking so we can overwrite the first state.
        mask_for_call = mask_override
        if x_init_noisy is not None and mask_for_call is None:
            if torch.is_tensor(x_in):
                mask_for_call = torch.zeros((1, x_in.shape[1], 1), dtype=torch.bool, device=x_in.device)

        if num_steps is None and x_init_noisy is None:
            return model.reverse_samples(
                B,
                x_in,
                cond,
                intermediate_every=intermediate_every,
                mask_override=mask_for_call,
            )

        call_intermediate_every = (intermediate_every or 1) if num_steps else intermediate_every
        result = model.reverse_samples(
            B,
            x_in,
            cond,
            num_timesteps=target_steps if target_steps is not None else -1,
            intermediate_every=call_intermediate_every,
            mask_override=mask_for_call,
        )

        # Continuous sampler does not append the final state; store the last sample for truncation.
        processed = _process_sampling_result(result, x_init_noisy, num_steps or 0)
        return processed

    raise ValueError(f"Unsupported sampling mode '{mode}'. Expected 'cond' or 'cont'.")
