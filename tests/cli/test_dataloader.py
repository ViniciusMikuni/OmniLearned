import tempfile
from pathlib import Path

from typer.testing import CliRunner

from omnilearned.cli import app

runner = CliRunner()


def test_download(test_dataset_name, test_dataset_file_paths):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        result = runner.invoke(
            app, ["dataloader", "--dataset", test_dataset_name, "--folder", temp_dir]
        )
        assert result.exit_code == 0, f"Error: {result.stdout} {result.stderr}"

        # Assert the top exists
        assert (temp_dir / test_dataset_name).exists(), (
            f"Top dataset not found in {temp_dir}"
        )

        # Assert the dataset paths exist
        for path in test_dataset_file_paths:
            assert (temp_dir / path).exists(), (
                f"Dataset path {path} not found in {temp_dir}"
            )
