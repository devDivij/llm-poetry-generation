# Bengali Payar: generation and evaluation

Generate a two-line Bengali Payar couplet from an English meaning, then evaluate
its poetic structure and semantic similarity to that meaning. This package
combines a fine-tuned Phi-4 generator, a rule-based Payar analyzer, and a
fine-tuned BGE-M3 embedding model behind one Python interface.

```text
English meaning → Phi-4 + LoRA adapter → Bengali couplet
                                              │
                          ┌───────────────────┴─────────────────┐
                          ▼                                     ▼
                 Meter, rhyme, diction              English–Bengali similarity
                          └───────────────────┬─────────────────┘
                                              ▼
                                Couplet + both check results
```

## How it works

1. **Generate:** format the English meaning with the saved system prompt and
   Phi-4 chat template, then generate a Bengali response using a LoRA adapter.
2. **Check structure:** analyze the two lines for estimated mātrā counts, pause
   boundaries, rhyme, readability, and recognized colloquial diction.
3. **Compare meaning:** embed the English meaning and complete Bengali couplet
   with the fine-tuned BGE-M3 model and calculate their cosine similarity.

The pipeline returns these results together. It does not automatically rewrite
failed outputs or combine the checks into a single acceptance score.

## Models

| Model | Role | Local directory used below |
|---|---|---|
| [eigensmith/phi4-lora-bengali-poetry-0shot](https://huggingface.co/eigensmith/phi4-lora-bengali-poetry-0shot) | Phi-4 LoRA adapter; instructions without a worked example | `models/phi4-payar-0shot` |
| [eigensmith/phi4-lora-bengali-poetry-1shot](https://huggingface.co/eigensmith/phi4-lora-bengali-poetry-1shot) | Phi-4 LoRA adapter; instructions with one worked example | `models/phi4-payar-1shot` |
| [eigensmith/bge-m3-bengali-finetuned](https://huggingface.co/eigensmith/bge-m3-bengali-finetuned) | Full BGE-M3 embedding checkpoint, fine-tuned on 20k English–Bengali examples | `models/bge-m3-bengali-20k` |

**0-shot and 1-shot describe the prompt format.** Both generators are fine-tuned
models. The 1-shot prompt includes a worked example; the 0-shot prompt does not.
Use the matching adapter and `setting` when loading the pipeline. The exact
prompts are included in `payar/prompts.json`.

The Phi-4 repositories contain **LoRA adapters**, not complete base models.
Unsloth reads `adapter_config.json`, loads
[`unsloth/phi-4-unsloth-bnb-4bit`](https://huggingface.co/unsloth/phi-4-unsloth-bnb-4bit),
and attaches the adapter. Base weights may be downloaded into the Hugging Face
cache on first use. Fully offline inference also requires those base weights.
The BGE-M3 folder contains the full embedder, tokenizer, pooling, and normalization
configuration.

## Installation

Use Python 3.10 or later in a virtual environment with CUDA-compatible PyTorch
and an NVIDIA GPU supported by Unsloth. From the directory containing
`pyproject.toml`, install the package and model dependencies:

```bash
python3 -m pip install -e ".[generation,semantic]"
```

If the model dependencies are already installed in a compatible environment,
install only the package:

```bash
python3 -m pip install --no-deps -e .
```

Importing `payar` does not load models or initialize CUDA. Call
`PayarPipeline.from_pretrained()` before importing Transformers or
SentenceTransformers elsewhere in the process, so Unsloth can apply its patches
first. See the validation section for dependency compatibility limits.

## Download the models

Run from the package directory. Download one generator adapter and the embedder;
both adapters are needed only to use both prompt settings.

```bash
hf download eigensmith/phi4-lora-bengali-poetry-0shot --local-dir models/phi4-payar-0shot
hf download eigensmith/bge-m3-bengali-finetuned --local-dir models/bge-m3-bengali-20k
```

For the 1-shot generator:

```bash
hf download eigensmith/phi4-lora-bengali-poetry-1shot --local-dir models/phi4-payar-1shot
```

The `hf` command is supplied by `huggingface_hub`. If a repository is private,
authenticate first with `hf auth login` using an account with access. For
reproducible deployments, add `--revision MODEL_COMMIT_SHA` to each download.
Keep downloaded filenames and subdirectories unchanged. Local directory names
are independent of the Hugging Face repository names; the examples above match
`model_manifest.json`.

## Generate and evaluate

```python
from payar import PayarPipeline

# Load once and reuse for successive meanings.
pipeline = PayarPipeline.from_pretrained(
    adapter_path="models/phi4-payar-0shot",
    embedding_path="models/bge-m3-bengali-20k",
    setting="0S",
)

result = pipeline.generate("I bow at the feet of the sage.")
print(result["bengali"])
print(result["syntactic"])
print(result["semantic"])
```

For 1-shot generation, use `models/phi4-payar-1shot` and `setting="1S"`.
Phi-4 loads on the visible CUDA GPU. The embedder defaults to CPU to leave GPU
memory for generation; set `semantic_device="cuda"` if enough memory is available.
Paths are relative to the current working directory; absolute paths can be used
from another application.

Generation uses greedy decoding with a default budget of 300 new tokens and a
total context budget of 1536 tokens. Set `max_new_tokens=` on `generate()` to
change the output budget. Inputs exceeding the context budget raise an error
rather than being silently truncated.

## Evaluate an existing couplet

Use the same loaded pipeline without generating a new couplet:

```python
english = "The devotee remains at the Lord's feet, with devotion in the heart."
bengali = (
    "প্রভুর চরণে সদা রহিল যে জন।",
    "তাহার হৃদয়ে নিত্য প্রেমের সাধন॥",
)
result = pipeline.evaluate(english, bengali)
```

Couplets accept a tuple of two lines or text separated by newlines or Bengali
dandas. A `|` marks a pause **within** a line, not the boundary between two lines.

For evaluation without loading a generator, use the components directly:

```python
from payar import SemanticClassifier, analyze_couplet

semantic = SemanticClassifier.from_pretrained("models/bge-m3-bengali-20k")
structure = analyze_couplet(bengali)
meaning = semantic.classify(english, bengali)
```

Structural analysis alone uses the Python standard library and needs no weights.
Semantic scoring requires the `semantic` dependencies and the embedder.

## Understanding the results

| Result field | Meaning |
|---|---|
| `english` | Supplied English meaning |
| `bengali` | Generated or supplied Bengali text |
| `syntactic` | Structural diagnostics: `payar`, `task_valid`, line analysis, rhyme, diction, and failures |
| `semantic` | Cosine similarity, optional threshold decision, and token counts |

The structural analyzer estimates 14 mātrā per line and checks word/pause
boundaries after 8, 7, or 9 mātrā, in that preference order. It also checks rhyme
and readability. `task_valid` additionally requires no recognized colloquial
diction. These are spelling/pronunciation heuristics for the supported
Sadhu/medieval convention, not a general Bengali grammar parser.

`semantic["cosine_similarity"]` compares the complete English meaning with the
complete Bengali couplet. It is **not a probability of correctness**.
`semantic_match` is `None` by default. Supplying `threshold=` when loading the
pipeline enables a comparison against that user-selected cutoff; the package
does not provide a calibrated threshold.

A wrong number of lines returns a failed structural check with an `error`
message. Empty generated text has `semantic=None`. Passing structural checks
does not prove that the meaning is correct, and a high similarity score does not
prove that the couplet is valid Payar.

## Checkpoint selection and validation

`model_manifest.json` records checkpoint 250 for 0S (validation loss 0.333771) and
checkpoint 200 for 1S (validation loss 0.329301). Selection follows the lowest
recorded validation loss within each run; it does not establish which generator
produces better poems.

Run the interface tests with:

```bash
python3 -B -m unittest discover -s tests -v
```

The tests use simulated generation and encoding with real structural analysis
and cosine calculation. They cover the combined interface and malformed/empty
outputs; they do not establish end-to-end GPU inference compatibility.

The original generation environment used Unsloth/Zoo 2026.11.4, Torch
2.14.1+cu130, and Transformers 5.17.0. The embedder was saved with
SentenceTransformers 6.2.0, Torch 2.15.0+cu132, and Transformers 5.22.1. These are
provenance records from separate environments, not a validated combined
dependency lock.
