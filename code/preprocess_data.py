"""Functions for converting raw data into processed data."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from . import photometry_data_import as pdi
from . import pycontrol_data_import as cdi
from .rsync import Rsync_aligner, RsyncError
from .utility import save_json, save_multiindex_df


def preprocess_data(
    raw_data_dir,
    processed_data_dir,
    plots_dir,
    trial_events,
    trial_responses=None,
    session_ids=None,
    skip_processed=False,
):
    """Preprocess a dataset by applying the following steps:

    1. Identify which photometry data file corresponds to which pyControl data file by
    matching subject ID and finding the closest start times.

    2. Iterate over sessions loading, preprocessing and syncronising the data.  The processed
    data is organised as folder structure: processed_data_dir/subject_dir/session_dir.
    Each indiviual session directory contains the following files:

        session_info.json    : dictionary with summary information about the session.
        events.htsv          : table containing pyControl events and state entries and correponding times.
        trials.htsv          : table with rows corresponding to trials of the pyControl task.
        variables.json       : dictionary with values of all pyControl task variables at the end of the session.
        photometry_info.json : dictionary with photometry acqusition parameters.
        dlight.signal.npy    : numpy array with the preprocessed photometry signal.
        dlight.times.npy     : numpy array with the times of each photometry signal sample.

    All times in processed data files are in seconds since the start the pyControl session.

    Parameters:
        raw_data_dir : path to directory containing raw data.
        processed_data_dir : path to directory where processed data will be saved.
        plots_dir : path to directory where plots showing photometry preprocessing will be saved.
        trial_events : dict indicating which trial event times should be recorded in the trials dataframe
                       see docstring for _add_trial_times() for more information.
        trial_responses : dict indicating which trial response times should be recorded in the trials
                         dataframe, see docstring for _add_trial_times() for more information.
        session_ids : List of sessions to process, if None then all sessions in raw_data_dir will be processed.
        skip_processed : set True to skip any sessions which are already in processed_data_dir
    """
    pycontrol_dir = Path(raw_data_dir, "pyControl")
    photometry_dir = Path(raw_data_dir, "photometry")
    # Find corresponding pyControl and photometry files.
    pycontrol_fileinfo_df = _get_file_info(pycontrol_dir, ".tsv")
    photometry_fileinfo_df = _get_file_info(photometry_dir, ".ppd")
    session_filepaths_df = pycontrol_fileinfo_df.loc[:, ("subject", "datetime", "filepath")]
    session_filepaths_df.rename(columns={"filepath": "pycontrol_filepath"}, inplace=True)
    for subject in set(session_filepaths_df["subject"]):
        subj_photo_info = photometry_fileinfo_df[photometry_fileinfo_df["subject"] == subject]
        for i, row in session_filepaths_df.iterrows():
            if row.subject == subject:
                closest_ind = (subj_photo_info["datetime"] - row.datetime).abs().idxmin()
                photo_row = subj_photo_info.loc[closest_ind, :]
                if (row.datetime - photo_row.datetime).total_seconds() < 3600:
                    session_filepaths_df.loc[i, "photometry_filepath"] = photo_row.filepath
    session_filepaths_df.to_csv(Path(raw_data_dir, "session_filepaths.tsv"), sep="\t", index=False)
    # Iterate over subjects and sessions preprocessing data.
    no_sync_sessions = []
    photo_error_sessions = []
    photo_plots_dir = Path(plots_dir, "photometry_preprocessing")
    start_date = min(session_filepaths_df.datetime).date()
    subject_info_df = pd.read_csv(Path(raw_data_dir, "subject_info.tsv"), sep="\t", index_col="subject")
    photo_plots_dir.mkdir(exist_ok=True, parents=True)
    for subject in sorted(set(session_filepaths_df["subject"])):
        subject_dir = Path(processed_data_dir, subject)
        subject_sessions = session_filepaths_df[session_filepaths_df.subject == subject]
        subject_sessions = subject_sessions.sort_values("datetime").reset_index(drop=True)
        for i, row in subject_sessions.iterrows():
            session_id = row.pycontrol_filepath.stem
            if session_ids and session_id not in session_ids:  # Only process specified sessions.
                continue
            session_dir = Path(subject_dir, row.datetime.strftime("%Y-%m-%d-%H%M%S"))
            if skip_processed and session_dir.exists():
                continue  # Skip session as already processed.
            session_dir.mkdir(exist_ok=True, parents=True)
            # Process pyControl data.
            info_dict, variables_dict, events_df, trials_df, sync_pulse_times = _pycontrol_to_components(
                row.pycontrol_filepath, trial_events, trial_responses
            )
            info_dict["subject"] = info_dict.pop("subject_id")
            info_dict["session_id"] = session_id
            info_dict["day"] = (row.datetime.date() - start_date).days + 1
            info_dict["number"] = i + 1
            info_dict["genotype"] = subject_info_df.loc[subject, "genotype"]
            save_json(info_dict, Path(session_dir, "session_info.json"))
            save_json(variables_dict, Path(session_dir, "variables.json"))
            events_df.to_csv(Path(session_dir, "events.htsv"), sep="\t", index=False)
            save_multiindex_df(trials_df, Path(session_dir, "trials.htsv"))
            # Process photometry data.
            photo_data = pdi.import_ppd(row.photometry_filepath, low_pass=None, high_pass=None)
            try:
                signal = pdi.preprocess_data(photo_data, fig_path=Path(photo_plots_dir, f"{session_id}.png"))
            except pdi.PreprocessingError:
                photo_error_sessions.append(session_id)
            # Compute times of photometry samples in pyControl seconds.
            sync_pulse_inds = photo_data["pulse_inds_2"]
            try:
                aligner = Rsync_aligner(pulse_times_A=sync_pulse_times, pulse_times_B=sync_pulse_inds)
            except RsyncError:  # Sync pulses times in photometry and pycontrol files do not match.
                no_sync_sessions.append(session_id)
                continue
            sample_times = aligner.B_to_A(np.arange(len(signal)))
            np.save(Path(session_dir, "dlight.signal.npy"), signal.astype(np.float32))
            np.save(Path(session_dir, "dlight.times.npy"), sample_times.astype(np.float32))
            photo_info = {
                "mode": photo_data["mode"],
                "sampling_rate": photo_data["sampling_rate"],
                "LED_current": photo_data["LED_current"],
                "hemisphere": subject_info_df.loc[subject, "hemisphere"],
            }
            save_json(photo_info, Path(session_dir, "photometry_info.json"))
    # Print any error output.
    if no_sync_sessions:
        print(f"No sync found for sessions: {no_sync_sessions}")
        with Path(raw_data_dir, "no_sync_sessions.txt").open("w") as f:
            f.write(repr(no_sync_sessions))
    if photo_error_sessions:
        print(f"Error during photometry preprocessing for sessions: {photo_error_sessions}")
        with Path(raw_data_dir, "photometry_error_sessions.txt").open("w") as f:
            f.write(repr(photo_error_sessions))


def _get_file_info(data_dir, filetype):
    """Given a target data directory and file extension, make a dataframe containing the file paths,
    subject IDs and datetimes of the data files.
    Assumes file names are of format: 'subject_ID-datetime_string.filetype'"""
    filepaths = [filepath for filepath in data_dir.iterdir() if filepath.suffix == filetype]
    subjects = [filepath.stem.split("-")[0] for filepath in filepaths]
    datetime_strings = [filepath.stem.split("-", 1)[1] for filepath in filepaths]
    datetimes = [datetime.strptime(datetime_string, "%Y-%m-%d-%H%M%S") for datetime_string in datetime_strings]
    return pd.DataFrame({"filepath": filepaths, "subject": subjects, "datetime": datetimes})


def _pycontrol_to_components(pycontrol_filepath, trial_events=None, trial_responses=None, str2dict_func=None):
    """Load a pycontrol session file and split it into a set of simpler components. The trial_events and
    trial_responses arguments are used to specify the times of events that will be recorded for each
    trial, see _add_trial_times docstring for details.  The str2dict_func argument is used if trial
    variables were output as user formatted print strings rather than using the print_variables function,
    and should be a function which converts the printed string to a dictionary {var_name: var_value}.
    """
    session_df = cdi.session_dataframe(pycontrol_filepath)
    # Make info_dict
    info_rows = session_df[session_df.type == "info"]
    info_dict = dict(zip(info_rows.subtype, info_rows.content))
    # Make events_df
    events_df = session_df[session_df.type.isin(["state", "event"]) & (session_df.subtype != "sync")]
    events_df = events_df.drop(columns="subtype").reset_index(drop=True)
    events_df.rename(columns={"content": "name"}, inplace=True)
    # Make trials_df
    if str2dict_func:  # Trial variables are in user formatted strings.
        trial_rows = session_df[(session_df.type == "print") & (session_df.subtype == "user")]
        trials_df = pd.DataFrame([str2dict_func(s) for s in trial_rows.content])
    else:  # Trial variables are in json formatted strings generated by print_variables().
        trial_rows = session_df[(session_df.type == "variable") & (session_df.subtype == "print")]
        trials_df = pd.DataFrame([json.loads(v) for v in trial_rows.content])
    if trial_events:
        trials_df = _add_trial_times(events_df, trials_df, trial_events, trial_responses)
    # Make variables_dict.
    if pycontrol_filepath.suffix == ".tsv":  # pyControl .txt files don't have this info.
        vars_row = session_df.loc[(session_df.type == "variable") & (session_df.subtype == "run_end")]
        variables_dict = json.loads(vars_row["content"].values[0])
    # Get sync pulse times.
    sync_pulse_times = session_df.loc[session_df.subtype == "sync", "time"].to_numpy()
    return info_dict, variables_dict, events_df, trials_df, sync_pulse_times


def _add_trial_times(events_df, trials_df, trial_events, trial_responses=None):
    """Add the times of trial events to the trials_df dataframe. Trial events are important time-
    points that occur once each trial, which may be used e.g. for aligning neural activity during
    analysis. For example, in a reversal learning task they could be the time the subject initiated
    the trial and the time they made the choice.  A given trial event might correspond to different
    pyControl events or state entries on different trials, for example the time of the choice might
    be indicated by entry into either a reward or no reward state (see below). Trial event times are
    added to the trials_df Dataframe as MultiIndexed columns, such that the times of a trial event
    called 'choice' can be accessed using trials_df.times.choice

    Parameters:

        events_df: Dataframe of pyControl event/state entry times with one row per event/state entry.

        trials_df: Dataframe with one row per trial.

        trial_events: dict mapping the name of the trial event to the corresponding pyControl events
            or state entries.  The format of the trial_events argument is:

                trial_events = {trial_event: pyControl_event/events}

            For example the trial events argument for a reversal learning task might be:

                trial_events = {'initiation' : 'choice_state',
                                'choice'     : ('reward_left', 'reward_right', 'no_reward')}

            This specifies that two times should be recorded in each row of the trials_df dataframe,
            one called 'initiation' whose time is given by entry into the tasks 'choice_state' and
            annother called 'choice', whose time is given by entry into any one of the tasks
            'reward_left','reward_right' or 'no_reward' states. Each trial event must occur once and
            only once on each trial. For this reason pyControl state entries rather than events
            triggered by subject's behavioural responses are usually used.  If you need to record
            times of subject's actions that do not trigger state transitions, use the trial_responses
            argument.

        trial_responses: dict specifying subjects responses whose times should be recorded in trials_df.
            A response is defined as the first occurence of a given pyControl event following a specified
            trial_event. The format of the trial_responses argument is:

                trial_responses = {trial_response: {preceding_trial_event: pyControl_event/events}}

            For example on a Pavlovian conditioning task if we want to record the time of the subjects
            first magazine entry following stimulus onset, we could specify:

                trial_responses = {'response': {'stimulus_onset': 'magazine_entry'}}

            This specifies that we want to record the time of the first pyControl 'magazine_entry' event
            after each 'stimulus_onset' trial event (specified using the trial_events argument) and
            record it in trials_df as column trials_df.times.response.

    Returns:
        trials_df : Dataframe with one row per trial with trial event times added as new columns.
    """
    # Find trial event times.
    trial_events = {
        k: [v] if isinstance(v, str) else v for k, v in trial_events.items()
    }  # Convert single event names to lists.
    trial_times = {k: events_df.loc[events_df.name.isin(v), "time"].values for k, v in trial_events.items()}
    trial_times_df = pd.DataFrame({k: pd.Series(v) for k, v in trial_times.items()})[: len(trials_df)]
    # Find trial response times.
    if trial_responses:
        for trial_response, spec_dict in trial_responses.items():
            trial_event, pycontrol_events = next(iter(spec_dict.items()))
            pycontrol_events = [pycontrol_events] if isinstance(pycontrol_events, str) else pycontrol_events
            pyc_event_times = events_df.loc[events_df.name.isin(pycontrol_events), "time"].to_numpy()
            response_inds = np.searchsorted(pyc_event_times, trial_times[trial_event])
            response_times = pyc_event_times[response_inds[response_inds < len(pyc_event_times)]]
            trial_times_df[trial_response] = pd.Series(response_times)
    # Convert trial_df columns to MultiIndex and add times columns.
    trial_times_df.columns = pd.MultiIndex.from_product([["times"], trial_times_df.columns])
    trials_df.columns = pd.MultiIndex.from_product([trials_df.columns, [""]])
    trials_df = pd.concat([trials_df, trial_times_df], axis=1)
    return trials_df
