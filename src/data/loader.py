from pathlib import Path
import pandas as pd


class DataLoader:
    """
    Responsible for loading datasets.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        return pd.read_csv(self.file_path)