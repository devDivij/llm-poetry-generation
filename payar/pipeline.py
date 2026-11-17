"""Generate and evaluate one Bengali couplet through a single interface."""

import json
from pathlib import Path

from .semantic import SemanticClassifier
from .syntactic import analyze_couplet

USER_MESSAGE = "Meaning:\n{english}\n\nGenerate the Bengali verse.\n"


class PayarPipeline:
    """Load once, then call generate(english) or evaluate(english, bengali)."""

    def __init__(self, model, tokenizer, semantic, system_prompt, *, max_length=1536):
        self.model = model
        self.tokenizer = tokenizer
        self.semantic = semantic
        self.system_prompt = system_prompt
        self.max_length = max_length

    @classmethod
    def from_pretrained(cls, adapter_path, embedding_path, *, setting="0S",
                        semantic_device="cpu", threshold=None):
        """Load the Phi-4 adapter on the visible CUDA GPU, then the embedder.

        Call before importing Transformers/SentenceTransformers elsewhere:
        Unsloth needs to patch Transformers first. Base weights may be downloaded.
        The embedder defaults to CPU to leave GPU memory for Phi-4.
        """
        prompts = json.loads(Path(__file__).with_name("prompts.json").read_text(encoding="utf-8"))
        if setting not in prompts:
            raise ValueError("setting must be '0S' or '1S'")
        path = Path(adapter_path).expanduser().resolve()
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Adapter checkpoint not found: {path}")
        if not Path(embedding_path).expanduser().is_dir():
            raise FileNotFoundError(f"Embedding checkpoint not found: {embedding_path}")
        metadata = path / "selection.json"
        if metadata.is_file() and json.loads(metadata.read_text())["setting"] != setting:
            raise ValueError("The requested prompt setting does not match this adapter")

        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(path), max_seq_length=1536, load_in_4bit=True, dtype=None)
        tokenizer = get_chat_template(tokenizer, chat_template="phi-4")
        FastLanguageModel.for_inference(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        semantic = SemanticClassifier.from_pretrained(
            embedding_path, device=semantic_device, threshold=threshold)
        return cls(model, tokenizer, semantic, prompts[setting])

    def evaluate(self, english, bengali):
        """Return both checks for an existing couplet, without generating.

        A wrong line count is reported as a structural failure. Empty text has
        no semantic score. Similarity is not a probability or an acceptance rule.
        """
        if not isinstance(english, str) or not english.strip():
            raise ValueError("english must be nonempty text")
        try:
            syntactic = analyze_couplet(bengali)
        except ValueError as error:
            syntactic = {"payar": False, "task_valid": False, "error": str(error)}
        semantic = (None if isinstance(bengali, str) and not bengali.strip()
                    else self.semantic.classify(english, bengali))
        return {"english": english, "bengali": bengali,
                "syntactic": syntactic, "semantic": semantic}

    def generate(self, english, *, max_new_tokens=300):
        """Generate greedily and return the couplet with both checks."""
        if not isinstance(english, str) or not english.strip():
            raise ValueError("english must be nonempty text")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": USER_MESSAGE.format(english=english)},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        count = inputs["input_ids"].shape[1]
        if count + max_new_tokens > self.max_length:
            raise ValueError("Prompt plus generation budget exceeds the model context length")
        import torch

        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         do_sample=False, use_cache=True,
                                         pad_token_id=self.tokenizer.pad_token_id)
        bengali = self.tokenizer.decode(output[0, count:], skip_special_tokens=True).strip()
        return self.evaluate(english, bengali)
