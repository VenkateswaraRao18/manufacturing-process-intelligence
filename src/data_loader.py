import pandas as pd
import numpy as np
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent / "data" / "dec2jan12_mldata.csv"


class DataLoader:
    def __init__(self, filepath=DATA_PATH):
        self.filepath = filepath
        self.raw_df = None
        self.vehicle_config = None   # (n_vehicles, 512) binary config matrix
        self.vehicle_tasks = None    # (n_vehicles, n_tasks) binary task matrix
        self.config_cols = None
        self.task_cols = None        # filtered task list (min_task_support threshold)
        self.vehicle_ids = None

    def load(self):
        self.raw_df = pd.read_csv(self.filepath, sep="|", encoding="latin-1")
        print(f"Loaded: {self.raw_df.shape[0]:,} rows × {self.raw_df.shape[1]} columns")
        return self

    def clean(self):
        # Remove duplicate (vehicle, task) pairs — keep first occurrence
        before = len(self.raw_df)
        self.raw_df = self.raw_df.drop_duplicates(subset=["SPNR8", "CHARACTERISTIC"])
        after = len(self.raw_df)
        if before != after:
            print(f"Removed {before - after} duplicate (vehicle, task) rows")

        self.config_cols = [c for c in self.raw_df.columns
                            if c not in ("SPNR8", "CHARACTERISTIC", "CHARACTERSTIC_DESCRIPTION")]

        # Verify binary
        assert self.raw_df[self.config_cols].isin([0, 1]).all().all(), \
            "Config columns contain non-binary values"
        print(f"Config columns: {len(self.config_cols)}")
        return self

    def pivot(self, min_task_support: float = 0.05):
        """
        Build two vehicle-level matrices:
          vehicle_config : (n_vehicles, 512) — constant binary config per vehicle
          vehicle_tasks  : (n_vehicles, n_tasks) — binary, did vehicle perform each task?

        min_task_support: minimum fraction of vehicles a task must appear in to be kept
                          as a label (filters out ultra-rare tasks)
        """
        # Vehicle config — one row per vehicle (config is constant, so just take first)
        config_df = (
            self.raw_df.groupby("SPNR8")[self.config_cols]
            .first()
            .reset_index()
        )
        self.vehicle_ids = config_df["SPNR8"].values
        self.vehicle_config = config_df[self.config_cols].values.astype(np.int8)

        # Task matrix — pivot: vehicles × tasks
        task_pivot = (
            self.raw_df.assign(performed=1)
            .pivot_table(index="SPNR8", columns="CHARACTERISTIC",
                         values="performed", aggfunc="max", fill_value=0)
        )
        # Align rows to vehicle_ids order
        task_pivot = task_pivot.reindex(self.vehicle_ids)

        # Filter tasks by minimum support
        n_vehicles = len(self.vehicle_ids)
        task_freq = task_pivot.mean()
        common_tasks = task_freq[task_freq >= min_task_support].index
        self.task_cols = list(common_tasks)
        self.vehicle_tasks = task_pivot[self.task_cols].values.astype(np.int8)

        print(f"\nVehicle-level matrices built:")
        print(f"  Vehicles        : {n_vehicles:,}")
        print(f"  Config features : {self.vehicle_config.shape[1]}")
        print(f"  Total tasks     : {task_pivot.shape[1]:,}")
        print(f"  Tasks kept (>={min_task_support*100:.0f}% support) : {len(self.task_cols)}")
        print(f"  Task matrix     : {self.vehicle_tasks.shape}")
        return self

    def summarize(self):
        n_vehicles = len(self.vehicle_ids)
        tasks_per_vehicle = self.raw_df.groupby("SPNR8")["CHARACTERISTIC"].nunique()
        config_density = self.vehicle_config.mean(axis=1)

        print("\n=== DATASET SUMMARY ===")
        print(f"Total vehicles        : {n_vehicles:,}")
        print(f"Unique tasks (total)  : {self.raw_df['CHARACTERISTIC'].nunique():,}")
        print(f"Tasks per vehicle     : {tasks_per_vehicle.mean():.1f} avg "
              f"(min {tasks_per_vehicle.min()}, max {tasks_per_vehicle.max()})")
        print(f"Config feature density: {config_density.mean()*100:.1f}% avg features = 1")
        print(f"Label matrix density  : {self.vehicle_tasks.mean()*100:.2f}% of labels per vehicle")
        print(f"Label matrix (kept)   : {self.vehicle_tasks.shape[1]} tasks × {n_vehicles} vehicles")

    def get_task_descriptions(self):
        """Returns dict: CHARACTERISTIC -> CHARACTERSTIC_DESCRIPTION"""
        return (
            self.raw_df[["CHARACTERISTIC", "CHARACTERSTIC_DESCRIPTION"]]
            .drop_duplicates("CHARACTERISTIC")
            .set_index("CHARACTERISTIC")["CHARACTERSTIC_DESCRIPTION"]
            .to_dict()
        )
