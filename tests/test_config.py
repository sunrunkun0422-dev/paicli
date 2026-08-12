from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from paicli.config import Settings, load_dotenv


class SettingsTest(unittest.TestCase):
    def test_project_env_is_loaded_and_process_env_wins(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("GLM_API_KEY=project-key\nGLM_MODEL=project-model\n", encoding="utf-8")
            with patch("paicli.config.Path.home", return_value=root / "missing-home"):
                settings = Settings.load(root, environ={"GLM_MODEL": "process-model"})
            self.assertEqual("project-key", settings.env["GLM_API_KEY"])
            self.assertEqual("process-model", settings.env["GLM_MODEL"])
            self.assertEqual("glm", settings.provider)

    def test_dotenv_does_not_override_supplied_environment(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("A=file\nB='quoted'\n", encoding="utf-8")
            values = load_dotenv(path, {"A": "process"})
            self.assertEqual("process", values["A"])
            self.assertEqual("quoted", values["B"])


if __name__ == "__main__":
    unittest.main()
