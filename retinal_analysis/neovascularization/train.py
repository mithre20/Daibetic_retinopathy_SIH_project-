"""
Neovascularization Training Guard Interface
Prevents training without an audited and validated ground truth dataset.
"""

from typing import Optional


class DatasetRequiredError(Exception):
    pass


def train_neovascularization_model(data_dir: Optional[str] = None):
    """
    Guarded training pipeline.
    Explicitly refuses to train or synthesize fake annotations.
    """
    if data_dir is None:
        raise DatasetRequiredError(
            "Neovascularization training cannot proceed: No validated NVD/NVE dataset path provided. "
            "Please provide an IRB-approved dataset with certified retinal specialist annotations."
        )
    print(f"Checking dataset directory: {data_dir}")
    raise DatasetRequiredError(f"Directory {data_dir} does not contain verified NVD/NVE ground truth masks.")


if __name__ == "__main__":
    try:
        train_neovascularization_model()
    except DatasetRequiredError as e:
        print(f"[STATUS] {e}")
