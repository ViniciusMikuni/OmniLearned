import tempfile
import re

from typer.testing import CliRunner

from omnilearned.cli import app

runner = CliRunner()


def test_train(test_dataset_name, test_dataset_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(
            app,
            [
                "train",
                "--epoch",
                2,
                "--path",
                test_dataset_path,
                "--dataset",
                test_dataset_name,
                "--output_dir",
                temp_dir,
            ],
        )
        assert result.exit_code == 0, f"Error: {result.stdout} {result.stderr}"

        lines = result.stdout.split("\n")
        # Remove empty lines
        lines = [line for line in lines if line.strip()]

        assert len(lines) > 0, "No output produced: {result.stderr}"

        pattern = r"best loss: (\d+\.\d+)"
        match = re.search(pattern, lines[-1])

        assert match, (
            f"Training Complete did not complete successfully: {result.stdout} {result.stderr}"
        )
        best_loss = float(match.group(1))
        assert best_loss > 0 and best_loss < 1, (
            f"Best loss out of range: {result.stdout} {result.stderr}"
        )
