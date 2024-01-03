"""This script implements the same steps as the ipython notebook but without the additional explanatory material."""

# %% Imports
import numpy as np
import pandas as pd
import pylab as plt
from pathlib import Path
from pprint import pprint
import preprocessing.preprocess_data as pp
import analysis.analysis as an

# %% Paths
raw_data_dir = Path("..", "data", "raw")  # Folder where the raw data is located.
processed_data_dir = Path("..", "data", "processed")  # Folder where the processed data will be saved.
analysis_data_dir = Path("..", "data", "analysis")  # Folder where analysis data will be saved.
plots_dir = Path("..", "plots")  # Path where plots will be saved.

# %% Run preprocessing.

trial_events = (
    {  # Events whose times will be recorded in the trials dataframe with corresponding pyControl state names.
        "initiation": ("choice_state", "forced_choice_left", "forced_choice_right"),
        "choice": ("chose_left", "chose_right"),
    }
)

pp.preprocess_data(raw_data_dir, processed_data_dir, plots_dir, trial_events)

# %% Make analysis data

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
    """ "Compute variables that will be used in the analysis and return as dataframe with one row per trial."""
    # Make variable indicating whether each trials choice was contra- or ipsi-lateral to the fiber.
    contra_action = "poke_4" if session.photometry.hemisphere == "R" else "poke_6"
    contra_choice = session.trials_df.choice == contra_action
    # Make variables coding for the previous trials outcome on trials where the previous choice was the same/different from the current choice.
    same_choice = session.trials_df.choice == session.trials_df.choice.shift(
        1
    )  # True if choice is same as previous trial.
    prev_outcome = 2 * (session.trials_df.outcome.shift(1, fill_value=0.5) - 0.5)  # Previous trial outcome coded 1/-1.
    prev_outcome_same = (prev_outcome * same_choice).astype(int)  # Prev. outcome on trials where previous choice same.
    prev_outcome_diff = (prev_outcome * ~same_choice).astype(int)  # Prev. outcome on trials where previous choice diff.
    return pd.DataFrame(
        {
            "contra_choice": contra_choice,
            "prev_outcome_same": prev_outcome_same,
            "prev_outcome_diff": prev_outcome_diff,
        }
    )


for session in sessions:
    analysis_vars_df = make_analysis_variables_df(session)
    analysis_vars_df.to_parquet(Path(session.info.analysis_data_dir, "trials.analysis_variables.parquet"))

# %% Run analyses

sessions = an.load_sessions(Path("..", "data", "processed"))

sessions_df = an.make_multisession_dataframe(sessions)  # Make a dataframe containing data from all sessions.

an.plot_response(sessions_df, alignment="trial", hue="outcome", style="contra_choice", fig_no=1)

an.regression_analysis(sessions_df, formula="outcome + contra_choice + correct", alignment="trial", fig_no=2)
