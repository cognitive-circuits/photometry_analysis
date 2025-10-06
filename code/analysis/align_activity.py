"""Code for time warping neurophysiology data to align activity across trials. 

Copyright (c) Thomas Akam 2025. Licenced under the GNU General Public License v3.
"""

import numpy as np


def align_signals(
    signals,
    sample_times,
    trial_times,
    target_times,
    pre_win=0,
    post_win=0,
    fs_out=25,
    smooth_SD="auto",
):
    """
    Timewarp continuous signals to align event times on each trial to specified target
    event times. For each trial, input sample times are linearly time warped to align
    that trial's event times with the target times.  Activity is then evaluated at a set of
    regularly spaced timepoints relative to the target event times by Gaussian smoothing
    around output timepoints. The pre_win and post_win arguments can be used to specify
    time windows before the first and after the last alignment event on each trial to be
    included in the output signals.

    Arguments:
        signals      : Signals to be aligned, either 1D [n_samples] or 2D [n_signals, n_samples]
        sample_times : Times when the samples occured (seconds) [n_samples]
        trial_times  : Times of events used for alignment for each trial (seconds) [n_trials, n_events]
        target_times : Times of events used for alignment in output aligned trial (seconds) [n_events].
        pre_win      : Time window before first event to include in aligned signals (seconds).
        post_win     : Time window after last event to include in aligned signals (seconds).
        fs_out       : The sampling rate of the aligned output signals (Hz).
        smooth_SD    : Standard deviation (seconds) of Gaussian smoothing applied to output signals.
                       If set to 'auto', smooth_SD is set to 1/fs_out.
    Returns:
        aligned_signals : Array of trial aligned signals [n_trials, n_signals, n_timepoints]
                          or [n_trials, n_timepoints] if signals is a 1D array]
        t_out: Times of each output firing rate time point (seconds) [n_timepoints].
        min_max_stretch: Minimum and maximum stretch factor for each trial.  Used to exclude
                         trials which have extreme deviation from target timings [n_trials, 2]
    """
    assert not np.any(np.diff(trial_times, 1) < 0), "trial_times must be monotonically increasing"
    assert not np.any(np.diff(target_times) < 0), "target_times must be monotonically increasing"

    # Make 1D signals into 2D array.
    one_dim_signal = len(signals.shape) == 1
    if one_dim_signal:
        signals = signals[np.newaxis, :]

    if smooth_SD == "auto":
        smooth_SD = 1 / fs_out

    t_out = np.arange(target_times[0] - pre_win, target_times[-1] + post_win, 1 / fs_out)  # Output sample times.

    n_trials = trial_times.shape[0]
    n_signals = signals.shape[0]
    n_timepoints = len(t_out)

    # Add non-warped interval before and after first and last trial times to prevent
    # edge effects, include pre and post windows if specified.

    pad_len = smooth_SD * 4  # Extension to alignement interval to prevent edge effects.
    target_times = np.hstack([target_times[0] - pre_win - pad_len, target_times, target_times[-1] + post_win + pad_len])
    trial_times = np.hstack(
        [trial_times[:, 0, None] - pre_win - pad_len, trial_times, trial_times[:, -1, None] + post_win + pad_len]
    )

    # Compute inter-event intervals and stretch factors to align trial intervals to target intervals.

    target_deltas = np.diff(target_times)  # Duration of inter-event intervals for aligned signals (ms).
    trial_deltas = np.diff(trial_times, 1)  # Duration of inter-event intervals for each trial (ms).

    stretch_factors = target_deltas / trial_deltas  # Amount each interval of each trial must be stretched/squashed by.
    min_max_stretch = np.stack([np.min(stretch_factors, 1), np.max(stretch_factors, 1)]).T  # Trial min & max stretch.

    # Loop over trials computing aligned signals.

    aligned_signals = np.full([n_trials, n_signals, n_timepoints], np.nan)

    for tr in np.arange(n_trials):
        if trial_times[tr, 0] < sample_times[0]:
            continue  # This trial occured before signals started.
        if trial_times[tr, -1] > sample_times[-1]:
            break  # This and subsequent trials occured after signals finshed.

        # Linearly warp sample times to align inter-event intervals to target.
        trial_samples = (trial_times[tr, 0] <= sample_times) & (sample_times < trial_times[tr, -1])
        trial_signals = signals[:, trial_samples]
        trial_sample_times = sample_times[trial_samples]  # Trial sample times before warping
        aligned_sample_times = np.zeros(len(trial_sample_times))  # Trial sample times after warping
        for j in range(target_deltas.shape[0]):
            mask = (trial_times[tr, j] <= trial_sample_times) & (trial_sample_times < trial_times[tr, j + 1])
            aligned_sample_times[mask] = (trial_sample_times[mask] - trial_times[tr, j]) * (
                target_deltas[j] / trial_deltas[tr, j]
            ) + target_times[j]

        # Calculate aligned signals.
        aligned_signals[tr, :, :] = _resample_signals(trial_signals, aligned_sample_times, t_out, smooth_SD)

    if one_dim_signal:  # Drop singleton dimension from output.
        aligned_signals = aligned_signals[:, 0, :]

    return aligned_signals, t_out, min_max_stretch


def _resample_signals(signals, signal_sample_times, output_sample_times, smooth_SD):
    """
    For input signals defined at a set of signal_sample_times, generate output
    signals resampled at timepoints output_sample_times, by applying a Gaussian
    weighting of signal samples around each output timepoint with standard
    deviation smooth_SD.
    Arguments:
        signals: Signal samples [n_signals, n_samples]
        signal_sample_times: times of the signal samples [n_samples]
        output_sample_times: Timepoints at which to evalutate the resampled signal [n_output_samples]
        smooth_SD: Standard deviation of Gaussian smoothing.
    Returns:
        output_samples[n_signals, n_output_samples]
    """
    # Compute Gaussian weights between all output and input times
    time_diffs = output_sample_times[:, None] - signal_sample_times[None, :]
    weights = np.exp(-0.5 * (time_diffs / smooth_SD) ** 2)
    # Normalize weights so they sum to 1 for each output sample
    weights /= np.sum(weights, axis=1, keepdims=True) + 1e-12  # avoid divide by zero
    # Apply weights to signals
    output_samples = signals @ weights.T  # (n_signals, n_output)
    return output_samples
