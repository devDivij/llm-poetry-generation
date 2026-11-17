# Bengali Payar pipeline

One interface generates a Bengali couplet from an English meaning and returns
meter/rhyme/diction diagnostics plus English–Bengali cosine similarity.
The semantic and syntactic implementations are copied from the existing tested
modules. Training, CSV processing, batch jobs, and optimizer states are excluded.

## Use

Install the code into your existing inference environment:

```bash
cd payar_pipeline
python3 -m pip install --no-deps -e .
```

Generation requires CUDA, PyTorch, Unsloth, Transformers, and their dependencies.
The saved training logs used Unsloth/Zoo 2026.11.4, Torch 2.14.1+cu130, and
Transformers 5.17.0. The embedding checkpoint was saved with SentenceTransformers
6.2.0, Torch 2.15.0+cu132, and Transformers 5.22.1. These describe two original
environments; they are not a validated combined dependency lock.

Optional extras declare the direct dependencies (`.[semantic]`, `.[generation]`).
Do not upgrade a working server environment just to install the code. A combined
GPU inference run still needs validation in a compatible environment.

Run from this directory, or use absolute model paths:

```python
from payar import PayarPipeline

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

Load once and reuse `pipeline.generate()` for successive inputs. For 1S, use
`models/phi4-payar-1shot` and `setting="1S"`. The exact system prompts are stored
in `payar/prompts.json`; the user message and Phi-4 chat template match training.
Load this pipeline before importing Transformers/SentenceTransformers elsewhere,
so Unsloth can apply its patches first. The embedder defaults to CPU to leave GPU
memory for Phi-4; set `semantic_device="cuda"` if sufficient memory is available.

To check an existing couplet with the same loaded pipeline:

```python
result = pipeline.evaluate(english, (first_bengali_line, second_bengali_line))
```

Couplets accept a tuple of two lines or text separated by newlines/Bengali dandas.
A `|` marks a pause within a line, not the boundary between two lines.
Wrong line counts return a failed syntactic check with an error message; empty
generated text has `semantic=None`. Overlong inputs raise errors rather than
silently truncating. Generation uses greedy decoding with a default budget of
300 new tokens and a total context budget of 1536.

Cosine similarity is not a probability. `semantic_match` remains `None` unless
you explicitly supply `threshold=` at loading. A calibrated cutoff, combined
acceptance decision, and automatic regeneration are not implemented. Meter and
diction checks are pronunciation/spelling heuristics, not a general grammar
parser. Passing them does not prove semantic correctness.

You can also import `SemanticClassifier` or `analyze_couplet` directly from
`payar`, without loading Phi-4. Importing the package itself loads no ML libraries.

## Models

| Local folder | Contents | Selection |
|---|---|---|
| `models/bge-m3-bengali-20k` | Full fine-tuned embedder, tokenizer, pooling and normalization configs | Existing 20k-example model |
| `models/phi4-payar-0shot` | LoRA adapter and tokenizer | Checkpoint 250, validation loss 0.333771 |
| `models/phi4-payar-1shot` | LoRA adapter and tokenizer | Checkpoint 200, validation loss 0.329301 |

Adapter selection follows the lowest recorded validation loss within each run.
It does not establish which model produces better poems. Provenance is recorded
in `model_manifest.json` and each adapter's `selection.json`.

The adapters need base model `unsloth/phi-4-unsloth-bnb-4bit`; base weights are not
bundled and may be downloaded into the Hugging Face cache on first use. The
embedding checkpoint is loaded locally. This is not a fully offline Phi-4 bundle.

Local model files use hard links to the originals to avoid duplicating about
3.3 GB on this nearly full filesystem. They behave as regular files for loading,
copying, and uploading. Treat their contents as immutable: editing a linked file
in place also changes the original. No original files were moved or removed.

## GitHub and Hugging Face

Commit this directory's code, prompts, documentation, tests, and manifest to
GitHub. `models/` is excluded by `.gitignore`. No repositories have been created
and no files have been uploaded.

Upload each model folder to its own Hugging Face model repository, for example:

```text
YOUR_ACCOUNT/bge-m3-bengali-20k
YOUR_ACCOUNT/phi4-payar-0shot
YOUR_ACCOUNT/phi4-payar-1shot
```

Change repository/folder names, not loader filenames such as `model.safetensors`,
`adapter_model.safetensors`, `modules.json`, or the pooling module directories.
Keep each folder's tokenizer and configuration files with its weights. The
adapter repositories also need the matching prompt; include the code repository
link and `payar/prompts.json` when preparing their final model cards.

After uploading, another machine can restore each folder with:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="YOUR_ACCOUNT/bge-m3-bengali-20k",
    revision="YOUR_MODEL_COMMIT_SHA",
    local_dir="models/bge-m3-bengali-20k",
)
```

Repeat for the two adapter repositories. Record the real repository IDs and
commit SHAs in the manifest when published. Hugging Face references:
[uploading models](https://huggingface.co/docs/hub/models-uploading),
[downloading models](https://huggingface.co/docs/huggingface_hub/en/guides/download).

## Verification

```bash
python3 -B -m unittest discover -s tests -v
```

These tests check the combined interface with simulated generation and encoding,
real syntactic analysis and cosine calculation, and malformed/empty output cases.
They do not validate real GPU inference or the joint ML dependency environment.
