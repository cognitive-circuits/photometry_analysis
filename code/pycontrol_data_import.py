"""Code for importing pyControl datafiles."""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime


def session_dataframe(file_path):
    """Generate a pandas dataframe from a pyControl data file containing the
    sessions data.  For .tsv files (pyControl version >=2.0) the dataframe
    is simply generated using pandas.read_csv. For .txt files (pyControl
    version < 2.0) the dataframe generated matches those generated from tsv
    files as closely as possible given the reduced information present in
    the .txt file format. The resulting dataframe has  columns:

    time : Time in seconds relative to the session start.
    type : Whether the row contains session 'info', a 'state' entry,
          'event', 'print' line, 'variable' values, 'error's or 'warnings's.
    subtype: The operation that generated the row, see pyControl docs.
    content : The name of the state, event or session information in the row,
              or value of variables and print lines.
    """

    file_path = Path(file_path)

    print(f"Importing data file: {file_path.stem}")

    if file_path.suffix == ".tsv":  # Load data from .tsv file.
        df = pd.read_csv(file_path, delimiter="\t")

    elif file_path.suffix == ".txt":  # Load data from .txt. file.
        with open(file_path, "r") as f:
            all_lines = [line.strip() for line in f.readlines() if line.strip()]

        # Make dataframe.
        state_IDs = eval(next(line for line in all_lines if line[0] == "S")[2:])
        event_IDs = eval(next(line for line in all_lines if line[0] == "E")[2:])
        ID2name = {v: k for k, v in {**state_IDs, **event_IDs}.items()}

        line_dicts = []
        for line in all_lines:
            if line[0] == "I":  # Info line.
                name, value = line[2:].split(" : ")
                # Make info lines consistent with .tsv files.
                name = name.lower().replace(" ", "_")
                if name == "start_date":
                    name = "start_time"
                    value = datetime.strptime(value, "%Y/%m/%d %H:%M:%S").isoformat()
                line_dicts.append({"time": 0, "type": "info", "subtype": name, "content": value})
            elif line[0] == "D":  # Data line.
                timestamp, ID = [int(i) for i in line.split(" ")[1:]]
                line_dicts.append(
                    {
                        "time": timestamp / 1000,
                        "type": "state" if ID in state_IDs.values() else "event",
                        "content": ID2name[ID],
                    }
                )
            elif line[0] == "P":  # Print line.
                time_str, print_str = line[2:].split(" ", 1)
                timestamp = int(time_str)
                try:  # print_variables output.
                    value_dict = json.loads(print_str)
                    line_dicts.append(
                        {
                            "time": timestamp / 1000,
                            "type": "variable",
                            "subtype": "print",
                            "content": value_dict,
                        }
                    )
                except json.JSONDecodeError:  # User print string.
                    line_dicts.append(
                        {
                            "time": timestamp / 1000,
                            "type": "print",
                            "subtype": "user",
                            "content": print_str,
                        }
                    )

        df = pd.DataFrame(line_dicts)
        df.reset_index(drop=True)
        df = df.reindex(columns=["time", "type", "subtype", "content"])

    return df
