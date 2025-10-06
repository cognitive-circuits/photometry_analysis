"""This script implements the same steps as the ipython notebook using the code organised into modules in the code/analysis folder."""

# %% Imports
import pandas as pd
from pathlib import Path
import analysis.analysis as an

# %% Paths
processed_data_dir = Path("..", "data", "processed")  # Processed data folder.
analysis_data_dir = Path("..", "data", "analysis")  # Folder where analysis data will be saved.

# %% Make analysis data.

sessions = an.load_sessions(processed_data_dir)

# Compute the median trial times that will be used for time warping.
median_trial_times = an.get_median_trial_times(
    sessions, trial_events=["initiation", "choice"], save_dir=analysis_data_dir
)

# Generate the trial aligned activity and save it out to the analysis data folder.
an.save_aligned_signals(
    sessions, analysis_data_dir, trial_events=["initiation", "choice"], target_event_times=median_trial_times
)


def make_analysis_variables_df(session):
    """Make dataframe containing some additional trial variables that will be used for analyses."""
    # contra_choice: True if chosen option is contralateral to recording hemisphere
    contra_action = "poke_4" if session.photometry.hemisphere == "R" else "poke_6"
    contra_choice = session.trials_df.choice == contra_action
    # same_choice:True if choice is same as previous trial.
    same_choice = session.trials_df.choice == session.trials_df.choice.shift(1)
    # prev_outcome: Previous trial outcome coded 1/-1 for reward/omission.
    prev_outcome = 2 * (session.trials_df.outcome.shift(1, fill_value=0.5) - 0.5)
    # prev_outcome_same: Previous trial outcome coded 1/-1 on trials where choice is same as previous trial, else 0
    prev_outcome_same = (prev_outcome * same_choice).astype(int)
    # prev_outcome_diff: Previous trial outcome coded 1/-1 on trials where choice is different from previous trial, else 0
    prev_outcome_diff = (prev_outcome * ~same_choice).astype(int)
    return pd.DataFrame(
        {"contra_choice": contra_choice, "prev_outcome_same": prev_outcome_same, "prev_outcome_diff": prev_outcome_diff}
    )


for session in sessions:
    analysis_vars_df = make_analysis_variables_df(session)
    analysis_vars_df.to_parquet(Path(session.info.analysis_data_dir, "trials.analysis_variables.parquet"))

# %% Run analyses

# sessions = an.load_sessions(Path("..", "data", "processed"))

sessions_df = an.make_multisession_dataframe(sessions)  # Make a dataframe containing data from all sessions.

an.plot_response(sessions_df, alignment="choice", hue="outcome", style="contra_choice", fig_no=1)

an.regression_analysis(sessions_df, formula="outcome + contra_choice + correct", alignment="trial", fig_no=2)
