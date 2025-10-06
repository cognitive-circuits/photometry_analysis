from pathlib import Path
import numpy as np
import pandas as pd
import pylab as plt
import warnings
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp, false_discovery_control
from statsmodels.regression.mixed_linear_model import MixedLMParams
from tqdm import tqdm

from .session import Session
from .align_activity import align_signals
from .utility import save_json

# Set default plotting parameters.
plt.rcParams["pdf.fonttype"] = 42
plt.rc("axes.spines", top=False, right=False)

# Turn off Pandas performance warnings as they are over-zealous when working with MultiIndex columns.
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# --------------------------------------------------------------------------------------------------
# Generate analysis data
# --------------------------------------------------------------------------------------------------


def load_sessions(processed_data_dir):
    """Load all sessions in directory with structure processed_data_dir/subject_dir/session_dir"""
    sessions = []
    for subject_dir in Path(processed_data_dir).iterdir():
        for session_dir in subject_dir.iterdir():
            sessions.append(Session(session_dir))
    return sessions


def get_median_trial_times(sessions, trial_events, save_dir=None):
    """Load the sessions in processed_data_dir and compute the median intervals between the events
    specified in trial_events, then save out the median timing of the trial events relative to the
    first event as a json in the analysis_data_dir.  These median trial timings are used for time
    warping trials to align activity across trials.
    """
    trials_df = pd.concat([session.trials_df.loc[:, "times"][trial_events] for session in sessions])
    median_trial_times = trials_df.diff(axis=1).median().to_dict()
    median_trial_times[trial_events[0]] = 0
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    save_json(median_trial_times, Path(save_dir, "median_trial_times.json"))
    return median_trial_times


def save_aligned_signals(
    sessions, analysis_data_dir, trial_events, target_event_times=None, window_dur=[-1, 2], skip_processed=False
):
    """For all sessions in processed_data_dir generate event aligned photometry signal dataframe
    for specified trial_events and  save to analysis_data_dir as parquet files.
    """
    for session in sessions:
        print(f"Processing session: {session.info.session_id}")
        if skip_processed and session.info.analysis_data_dir.exists():
            continue  # Skip session as already processed.
        if session.photometry == None:
            continue  # Skip as session does not have photometry data.
        aligned_signal_dfs = []
        # Extract event aligned signals.
        for trial_event in trial_events:
            aligned_signal_dfs.append(get_event_aligned_signal(session, trial_event))
        # Extract time-warped trial aligned signals.
        if target_event_times:
            aligned_signal_dfs.append(get_trial_aligned_signal(session, target_event_times, window_dur))
        session.info.analysis_data_dir.mkdir(exist_ok=True, parents=True)
        aligned_signals_df = pd.concat(aligned_signal_dfs, axis=1)
        aligned_signals_df.to_parquet(Path(session.info.analysis_data_dir, "trials.aligned_signal.parquet"))


def get_event_aligned_signal(session, trial_event, window_dur=[-1, 2]):
    """Extract the photometry signal on each trial around a specfied event and return as a
    Dataframe with MultiIndex columns where the 1st level is the event name and the 2nd level
    are times in seconds relative to the event.
    Arguments:
        session: Session object to be processed
        trial_event: Name of the event to extract signals around.
        window_dur = Duation of the window around the event specifing number of seconds before and after.
    Returns:
        aligned_signal_df: Dataframe with one row per trial containing the aligned signal.
    """
    window_len = (np.array(window_dur) * session.photometry.sampling_rate).astype(int)
    window_times = np.arange(*window_dur, 1 / session.photometry.sampling_rate)
    event_times = session.trials_df.times[trial_event].to_numpy()
    event_inds = np.searchsorted(session.photometry.times, event_times)
    aligned_signal = np.full((len(event_times), len(window_times)), np.nan)
    for i, (event_time, event_ind) in enumerate(zip(event_times, event_inds)):
        if np.isnan(event_time):
            continue
        else:
            event_window = event_ind + window_len
            if (event_window[0] < 0) or event_window[1] > len(session.photometry.signal):
                continue
            else:
                aligned_signal[i, :] = session.photometry.signal[event_window[0] : event_window[1]]
    aligned_signal_df = pd.DataFrame(aligned_signal, columns=window_times)
    # Make Multiindex with event name and time in window as levels.
    aligned_signal_df.columns = pd.MultiIndex.from_product([[trial_event], aligned_signal_df.columns])
    aligned_signal_df.columns.names = ["event", "time"]
    return aligned_signal_df


def get_trial_aligned_signal(session, target_event_times, window_dur=[-1, 2], fs_out=50):
    """Compute time warped photometry signal for each trial by warping the intervals between trial
    events to match those specified in target_event_times.  Typically target_event_times will be
    based on the median inter-event intervals across the dataset. The aligned activity is returned
    as a Dataframe with MultiIndex columns where the 1st level is 'trial' and the 2nd level
    are times in seconds relative to the target event times.  For information on how the time
    warping is implemented see the docstring for align_activity.align_signals.
    Arguments:
        session: Session object to be processed
        target_event_times: Dict whose keys are event names and values are the times in seconds that
            the corresponding event should occur in the aligned data.  E.g. to specify that the
            'initiation' event should occur at 0 seconds and the 'choice' event should occur at 0.6
            seconds you would use: target_event_times = {'initiation':0, 'choice': 0.6}
        window_dur = Duation of the window around the event specifing number of seconds before and after.
    Returns:
        aligned_signal_df: Dataframe with one row per trial containing the aligned signal.
    """
    aligned_signals, t_out, _ = align_signals(
        signals=session.photometry.signal,
        sample_times=session.photometry.times,
        trial_times=session.trials_df.times.loc[:, target_event_times.keys()].to_numpy(),
        target_times=np.array(list(target_event_times.values())),
        pre_win=-window_dur[0],
        post_win=window_dur[1],
        fs_out=fs_out,
    )
    aligned_signal_df = pd.DataFrame(aligned_signals, columns=pd.MultiIndex.from_product([["trial"], t_out]))
    return aligned_signal_df


# --------------------------------------------------------------------------------------------------
# Analyse signals
# --------------------------------------------------------------------------------------------------


def make_multisession_dataframe(sessions):
    """Combine data from multiple sessions into a single dataframe."""
    session_dfs = []
    for session in sessions:
        session_df = session.get_analysis_df()
        # Add session-level variables as new columns.
        for info_name in ["genotype", "day", "session_id", "subject"]:
            session_df.insert(0, info_name, getattr(session.info, info_name))
        session_dfs.append(session_df)
    return pd.concat(session_dfs, axis=0)


def plot_response(sessions_df, alignment, hue, style=None, errorbar="se", fig_no=1):
    """Plot the response to a given trial event split by one or more conditions.  For each condition
    the cross subject mean is plotted, with cross subject variation shown as a shaded area.
    For example to plot activity aligned on subjects choice, split by trial outcome
    and genotype, with outcome indicated by line hue and genotype by line style:

        plot_response(sessions_df, alignment="choice", hue="outcome", style="genotype")

    Paramters:
        sessions_df : Dataframe generated by make_analysis_dataframe function.
        alignment: Which trial event to align to, e.g. "choice"
        hue: Grouping variable assigned to line hue.
        style: Grouping variable assigned to line style.
        errorbar: Metric to use for cross subject variation, can be "se", "sd", or other seaborn errorbar value.
        fig_no: Figure number of plot.
    """
    grouping = [hue, style] if style else [hue]
    # Calculate subjects mean trace for each condition.
    subject_means_df = sessions_df.groupby(["subject"] + grouping).mean(numeric_only=True)["aligned_signal", alignment]
    # Convert Dataframe to long form
    subject_means_ldf = subject_means_df.reset_index().melt(
        id_vars=["subject"] + grouping, var_name="time", value_name="signal"
    )
    # Plotting
    plt.figure(fig_no, figsize=[12, 8], clear=True)
    sns.lineplot(subject_means_ldf, x="time", y="signal", hue=hue, style=style, errorbar=errorbar)
    plt.xlim(subject_means_ldf.time.min(), subject_means_ldf.time.max())
    plt.xlabel(f"Time relative to {alignment} (s)")
    plt.ylabel("dLight dF/F")
    plt.legend(loc="upper left")


def regression_analysis(sessions_df, formula, alignment, style=None, sum_code=True, errorbar="se", fig_no=1):
    """Run a linear regression analysis predicting the photomety signal using predictors which vary from
    trial-to-trial.  A seperate regression analysis is run for each subject at each timepoint of the
    aligned activity.  The timecourse of the cross-subject mean beta for each predictor is plotted with
    cross subject variation indicated by shaded area.  The set of predictors used is specified using a
    string formatted according to the statsmodels formula api.
    Parameters:
        sessions_df: Dataframe generated by make_analysis_dataframe function.
        formula: String specifying the predictors to use in the regression.
        sum_code: If True binary variables are coded [-1,1] for orthogonal sum-to-zero contrasts.
        errorbar: Metric to use for cross subject variation, can be "se", "sd", or other seaborn errorbar value.
        fig_no: Figure number of plot.
    """
    timepoints = sessions_df.aligned_signal[alignment].columns
    coefs_dfs = []
    for subject in tqdm(sessions_df.subject.unique()):
        subject_df = sessions_df.loc[sessions_df.subject == subject, :]
        regression_df = subject_df.loc[:, [col for col in subject_df.columns if col[0] in formula]]
        regression_df.columns = regression_df.columns.droplevel([1, 2])
        if sum_code:
            regression_df.replace({True: 1, False: -1}, inplace=True)
        coefs = []
        for t in timepoints:
            regression_df["signal"] = subject_df.loc[:, ("aligned_signal", alignment, t)]
            fit = smf.ols(formula="signal ~ " + formula, data=regression_df).fit()
            coefs.append(fit.params)
        subject_coefs_df = pd.DataFrame(coefs)
        subject_coefs_df["time"] = timepoints
        subject_coefs_df["subject"] = subject
        if style:  # Group subjects by specified variable and differentiate by linestyle.
            assert len(subject_df[style].unique()) == 1, "Style variable must take unique value per subject."
            subject_coefs_df[style] = subject_df[style].unique()[0]
        coefs_dfs.append(subject_coefs_df)
    coefs_df = pd.concat(coefs_dfs)
    id_vars = ["subject", "time", style] if style else ["subject", "time"]
    coefs_ldf = coefs_df.melt(id_vars=id_vars, var_name="predictor", value_name="beta")
    # Compute predictor p values
    predictors = [col for col in coefs_df if col not in ("time", "subject", style)]
    pvals_df = pd.DataFrame({predictor: _predictor_pvalues(coefs_df, predictor) for predictor in predictors})
    pvals_df.index = timepoints
    # Plotting
    plt.figure(fig_no, figsize=[12, 8], clear=True)
    plt.axhline(0, color="k", linewidth=0.5)
    sns.lineplot(coefs_ldf, x="time", y="beta", hue="predictor", style=style, errorbar=errorbar)
    _plot_P_values(pvals_df, y0=plt.gca().get_ylim()[1])
    plt.xlim(coefs_ldf.time.min(), coefs_ldf.time.max())
    plt.xlabel(f"Time relative to {alignment} (s)")
    plt.ylabel("Beta (dLight dF/F)")
    plt.legend(loc="center left")


def _predictor_pvalues(coefs_df, predictor, multi_correct=True):
    """Compute P value for specified predictor at each timepoint using a ttest of the cross-subject
    distribution against 0.  If multi_correct is True Benjamini-Hochberg multiple comparison
    correction is applied to control false discovery rate."""
    predictor_df = coefs_df.loc[:, (predictor, "time", "subject")]
    predictor_df = predictor_df.pivot(columns=["time"], index="subject")
    p_values = ttest_1samp(predictor_df, 0, axis=0).pvalue
    if multi_correct:
        p_values = false_discovery_control(p_values)
    return p_values


def _plot_P_values(pvals_df, y0):
    """Indicate significance levels with dots of different sizes above plot."""
    t = pvals_df.index.to_numpy()
    for i, predictor in enumerate(pvals_df.columns):
        y = y0 * (1 + 0.04 * i)
        p_vals = pvals_df[predictor]
        t05 = t[(p_vals < 0.05) & (p_vals >= 0.01)]
        t01 = t[(p_vals < 0.01) & (p_vals >= 0.001)]
        t00 = t[p_vals < 0.001]
        plt.plot(t05, np.ones(t05.shape) * y, ".", markersize=3, color="C{}".format(i))
        plt.plot(t01, np.ones(t01.shape) * y, ".", markersize=6, color="C{}".format(i))
        plt.plot(t00, np.ones(t00.shape) * y, ".", markersize=9, color="C{}".format(i))


def mixed_effects_regression(
    sessions_df,
    formula="outcome*genotype",
    group="subject",
    re_formula="outcome",
    alignment="trial",
    sum_code=True,
    errorbar="se",
    downsample=1,
    fig_no=1,
):
    """Run a linear mixed effects regression analysis.  This is currently experimental and not recomended to use."""
    timepoints = sessions_df.aligned_signal[alignment].columns[::downsample]

    reg_df = sessions_df.loc[:, [col for col in sessions_df.columns if col[0] in "".join([formula, re_formula, group])]]
    reg_df.columns = reg_df.columns.droplevel([1, 2])
    if sum_code:
        reg_df.replace({True: 1, False: -1}, inplace=True)
        reg_df.replace({"A": 1, "B": -1}, inplace=True)
    coefs = []
    coefSEs = []
    for t in tqdm(timepoints):
        reg_df["signal"] = sessions_df.loc[:, ("aligned_signal", alignment, t)]
        md = smf.mixedlm("signal ~ " + formula, reg_df, re_formula=re_formula, groups=reg_df[group])
        fit = md.fit(free=MixedLMParams.from_components(np.ones(md.k_fe), np.eye(md.k_re)), method=["lbfgs"])
        coefs.append(fit.params[: md.k_fe])
        coefSEs.append(fit.bse_fe)
    coefs_df = pd.DataFrame(coefs)
    coefs_df["time"] = timepoints
    coefSE_df = pd.DataFrame(coefSEs)
    coefSE_df["time"] = timepoints
    # Plotting
    predictors = [col for col in coefs_df if col not in ("time", "subject")]
    plt.figure(fig_no, clear=True)
    plt.axhline(0, color="k", linewidth=0.5)
    palette = iter(sns.husl_palette(len(predictors)))
    for predictor in predictors:
        _plot_trace(
            coefs_df["time"].to_numpy(float),
            coefs_df[predictor],
            coefSE_df[predictor],
            color=next(palette),
            label=predictor,
        )
    plt.xlim(coefs_df.time.min(), coefs_df.time.max())
    plt.xlabel(f"Time relative to {alignment} (s)")
    plt.ylabel("Beta (dLight dF/F)")
    plt.legend(loc="upper left")


def _plot_trace(t, y, yerr, color=None, label=None):
    plt.plot(t, y, c=color)
    plt.fill_between(t, y - yerr, y + yerr, color=color, alpha=0.3, label=label)
