import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json_to_excel


class ChooseProviderTests(unittest.TestCase):
    @patch("builtins.input", return_value="")
    def test_empty_selection_uses_deepseek_default(self, _):
        self.assertEqual(json_to_excel.choose_provider(), "deepseek")

    @patch("builtins.input", return_value="requesty_qwen")
    def test_provider_name_can_be_selected(self, _):
        self.assertEqual(json_to_excel.choose_provider(), "requesty_qwen")

    @patch("builtins.input", side_effect=["invalid", "2"])
    def test_invalid_choice_prompts_again(self, _):
        self.assertEqual(json_to_excel.choose_provider(), "requesty_llama")


class ConvertToExcelTests(unittest.TestCase):
    def test_converts_selected_provider_stage_four_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            provider_dir = Path(temp_dir) / "requesty_qwen"
            provider_dir.mkdir()
            input_file = provider_dir / "stage_4_items.jsonl"
            rows = [
                {"item": "first", "confidence": 0.9},
                {"item": "second", "confidence": 0.8},
            ]
            input_file.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            output_file = json_to_excel.convert_to_excel(
                "requesty_qwen", outputs_dir=temp_dir
            )

            self.assertEqual(
                output_file, provider_dir / "eclass_final.xlsx"
            )
            self.assertEqual(pd.read_excel(output_file).to_dict("records"), rows)

    def test_missing_stage_four_output_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                FileNotFoundError, "requesty_llama"
            ):
                json_to_excel.convert_to_excel(
                    "requesty_llama", outputs_dir=temp_dir
                )


class MainTests(unittest.TestCase):
    @patch("json_to_excel.convert_to_excel")
    @patch("json_to_excel.choose_provider")
    def test_prompts_when_provider_is_not_passed(
        self, choose_provider, convert_to_excel
    ):
        choose_provider.return_value = "requesty_qwen"

        json_to_excel.main([])

        choose_provider.assert_called_once_with()
        convert_to_excel.assert_called_once_with("requesty_qwen")

    @patch("json_to_excel.convert_to_excel")
    @patch("json_to_excel.choose_provider")
    def test_provider_argument_skips_prompt(
        self, choose_provider, convert_to_excel
    ):
        json_to_excel.main(["--provider", "requesty_llama"])

        choose_provider.assert_not_called()
        convert_to_excel.assert_called_once_with("requesty_llama")


if __name__ == "__main__":
    unittest.main()
