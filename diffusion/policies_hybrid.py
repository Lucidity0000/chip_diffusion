"""
Hybrid policy helpers that mirror diffusion/policies.py but support optional warm-starts,
noise injection, and truncated trajectories via sampler_wrappers.run_sampling.
"""
import numpy as np
import torch

import utils
from diffusion.sampler_wrappers import run_sampling


def _parse_sampling_outputs(result):
    """Extract (samples, intermediates, extras) from run_sampling output."""
    if isinstance(result, (tuple, list)) and result:
        samples = result[0]
        intermediates = result[1] if len(result) > 1 and isinstance(result[1], (list, tuple)) else []
        extras = result[2:] if len(result) > 2 else ()
    else:
        samples = result
        intermediates = []
        extras = ()
    return samples, list(intermediates), extras


def diffusion_move(
    model,
    x_cur,
    cond,
    *,
    mode: str = "cond",
    x_init=None,
    noise_level: float = 0.0,
    num_steps: int = None,
    intermediate_every: int = 0,
    mask_override=None,
):
    """
    Single reverse-sampling step (useful for MCMC-like moves).
    Returns: samples, intermediates
    """
    result = run_sampling(
        model,
        x_cur,
        cond,
        mode=mode,
        num_steps=num_steps,
        x_init=x_init if x_init is not None else x_cur,
        noise_level=noise_level,
        intermediate_every=intermediate_every,
        mask_override=mask_override,
    )
    samples, intermediates, _ = _parse_sampling_outputs(result)
    return samples, intermediates


def open_loop(
    batch_size,
    model,
    x_in,
    cond,
    intermediate_every: int = 200,
    save_videos: bool = False,
    fps: int = 60,
    *,
    mode: str = "cond",
    x_init=None,
    noise_level: float = 0.0,
    num_steps: int = None,
    mask_override=None,
):
    """
    Default sampling with optional warm-starts/noise/truncation.
    """
    frame_output_rate = 2 if save_videos else intermediate_every
    sampling_result = run_sampling(
        model,
        x_in,
        cond,
        mode=mode,
        num_steps=num_steps,
        x_init=x_init,
        noise_level=noise_level,
        intermediate_every=frame_output_rate,
        mask_override=mask_override,
    )
    samples, intermediates, _ = _parse_sampling_outputs(sampling_result)
    intermediates = list(intermediates)
    intermediates.append(samples)

    metrics_special = {}
    if save_videos:
        frames = [utils.visualize_placement(intermediate.squeeze(dim=0), cond) for intermediate in intermediates] # T, H, W, C
        frames.extend([frames[-1]] * fps)
        log_video = np.moveaxis(np.array(frames), -1, 1) # T, C, H, W
        metrics_special["diffusion_video"] = utils.logging_video(log_video, fps=fps)
    return samples, intermediates, metrics_special


def open_loop_clustered(
    batch_size,
    model,
    x_in,
    cond,
    intermediate_every: int = 200,
    *,
    mode: str = "cond",
    x_init=None,
    noise_level: float = 0.0,
    num_steps: int = None,
    mask_override=None,
):
    """
    Clustered sampling variant with optional warm-starts/noise/truncation.
    """
    cluster_cond, cluster_x = utils.cluster(cond, num_clusters=512, placements=x_in, verbose=False)
    cluster_x_init = None
    if x_init is not None:
        _, cluster_x_init = utils.cluster(cond, num_clusters=512, placements=x_init, verbose=False)

    sampling_result = run_sampling(
        model,
        cluster_x,
        cluster_cond,
        mode=mode,
        num_steps=num_steps,
        x_init=cluster_x_init,
        noise_level=noise_level,
        intermediate_every=intermediate_every,
        mask_override=mask_override,
    )
    cluster_samples, cluster_intermediates, _ = _parse_sampling_outputs(sampling_result)
    samples = utils.uncluster(cluster_cond, cluster_samples)
    intermediates = [utils.uncluster(cluster_cond, intermediate) for intermediate in cluster_intermediates]
    return samples, intermediates
