"""Integration checks without downloading models or requiring a GPU."""

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from payar import PayarPipeline, SemanticClassifier

COUPLET = "প্রভুর চরণে সদা রহিল যে জন।\nতাহার হৃদয়ে নিত্য প্রেমের সাধন॥"


class Encoder:
    max_seq_length = 128

    def tokenizer(self, text, **kwargs):
        return {"input_ids": text.split()}

    def encode(self, texts, **kwargs):
        return [[3, 0], [3, 4]]


class Inputs(dict):
    def __init__(self):
        super().__init__(input_ids=SimpleNamespace(shape=(1, 10)))

    def to(self, device):
        return self


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.model = Mock(device="cuda:0")
        self.tokenizer = Mock(return_value=Inputs(), pad_token_id=0)
        self.tokenizer.decode.return_value = COUPLET
        self.output = Mock()
        self.output.__getitem__ = Mock(return_value=[100, 101])
        self.model.generate.return_value = self.output
        self.pipeline = PayarPipeline(self.model, self.tokenizer,
                                      SemanticClassifier(Encoder()), "original prompt")

    def test_generate_returns_both_real_checks_and_decodes_only_response(self):
        with patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}):
            result = self.pipeline.generate("English meaning")
        self.assertEqual(result["bengali"], COUPLET)
        self.assertTrue(result["syntactic"]["payar"])
        self.assertAlmostEqual(result["semantic"]["cosine_similarity"], 0.6)
        self.assertIsNone(result["semantic"]["semantic_match"])
        self.output.__getitem__.assert_called_once_with((0, slice(10, None)))
        self.assertFalse(self.model.generate.call_args.kwargs["do_sample"])
        messages = self.tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0]["content"], "original prompt")
        self.assertEqual(messages[1]["content"], "Meaning:\nEnglish meaning\n\nGenerate the Bengali verse.\n")
        json.dumps(result, ensure_ascii=False)

    def test_existing_tuple_is_checked_without_generation(self):
        result = self.pipeline.evaluate("English meaning", tuple(COUPLET.splitlines()))
        self.assertTrue(result["syntactic"]["payar"])
        self.model.generate.assert_not_called()

    def test_malformed_generated_text_is_reported(self):
        result = self.pipeline.evaluate("meaning", "শুধু একটি পঙক্তি")
        self.assertFalse(result["syntactic"]["payar"])
        self.assertIn("error", result["syntactic"])
        self.assertIsNotNone(result["semantic"])

    def test_empty_response_has_no_score(self):
        result = self.pipeline.evaluate("meaning", "")
        self.assertFalse(result["syntactic"]["task_valid"])
        self.assertIsNone(result["semantic"])

    def test_context_overflow_rejected_before_generation(self):
        with self.assertRaisesRegex(ValueError, "context length"):
            self.pipeline.generate("meaning", max_new_tokens=1530)
        self.model.generate.assert_not_called()

    def test_invalid_generation_arguments(self):
        for text in ("", " ", None):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.pipeline.generate(text)
        for budget in (True, 0, -1, 2.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.pipeline.generate("meaning", max_new_tokens=budget)
        with self.assertRaises(ValueError):
            self.pipeline.evaluate("", "")

    def test_import_has_no_model_or_gpu_side_effects(self):
        package_root = str(Path(__file__).resolve().parents[1])
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import payar; "
                "assert not {'torch', 'unsloth', 'sentence_transformers'} & sys.modules.keys()")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, package_root],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
