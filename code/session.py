"""Classes for loading and representing processed data."""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from .utility import load_json, load_multiindex_df


@dataclass
class Session_info:
    """Dataclass for representing session info."""

    path: str
    analysis_data_dir: Path
    session_id: str
    experiment_name: str
    task_name: str
    task_file_hash: str
    setup_id: str
    framework_version: str
    micropython_version: str
    subject: str
    start_time: datetime
    end_time: datetime
    day: int
    number: int
    genotype: str = None

    def __post_init__(self):
        self.start_time = datetime.fromisoformat(self.start_time)
        self.end_time = datetime.fromisoformat(self.end_time)


@dataclass
class Photometry:
    """Dataclass for representing photometry data and info."""

    mode: str
    sampling_rate: int
    LED_current: list
    hemisphere: str
    signal: np.ndarray
    times: np.ndarray
    aligned_signal_df: pd.DataFrame = None


class Session:
    """Class for loading and representing processed data from one session."""

    def __init__(self, data_dir):
        print(f"Loading session: {data_dir}")
        analysis_dir = Path(*[part if part != "processed" else "analysis" for part in data_dir.parts])
        self.info = Session_info(data_dir, analysis_dir, **load_json(Path(data_dir, "session_info.json")))
        self.variables = load_json(Path(data_dir, "variables.json"))
        self.events_df = pd.read_csv(Path(data_dir, "events.htsv"), sep="\t")
        self.trials_df = load_multiindex_df(Path(data_dir, "trials.htsv"))
        # Load photometry signals if available.
        if Path(data_dir, "dlight.signal.npy").exists():
            self.photometry = Photometry(
                signal=np.load(Path(data_dir, "dlight.signal.npy")),
                times=np.load(Path(data_dir, "dlight.times.npy")),
                **load_json(Path(data_dir, "photometry_info.json")),
            )
            if Path(analysis_dir, "trials.aligned_signal.parquet").exists():
                self.photometry.aligned_signal_df = pd.read_parquet(Path(analysis_dir, "trials.aligned_signal.parquet"))
        else:
            self.photometry = None

    def get_trial_aligned_signal(self):
        """Return a dataframe with trial_df concatentated to aligned_signal_df."""
        trials_df = self.trials_df.copy()
        signals_df = self.photometry.aligned_signal_df.copy()
        # Convert both DataFrames to 3 level MultiIndex.
        trials_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in trials_df.columns])
        signals_df.columns = pd.MultiIndex.from_tuples([("aligned_signal", *col) for col in signals_df.columns])
        return pd.concat([trials_df, signals_df], axis=1)
