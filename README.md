# Part-of-Speech Taggers

A PyTorch implementation of three part-of-speech (POS) taggers with increasing sophistication: a **Hidden Markov Model (HMM)**, a **Linear-Chain Conditional Random Field (CRF)**, and a **CRF with Bidirectional RNN features (CRF-BiRNN)**. The project supports both supervised and semi-supervised training, and includes evaluation on English, Czech, and a toy ice-cream dataset.

## Models

| Model | File | Description |
|-------|------|-------------|
| HMM | `code/hmm.py` | First-order Hidden Markov Model with vectorized Viterbi decoding, forward algorithm (supervised & unsupervised), and SGD training with early stopping. Supports unigram mode and semi-supervised learning via marginalization. |
| CRF | `code/crf.py` | Linear-chain Conditional Random Field using exp-based potentials (proper CRF parameterization). Computes conditional log-likelihood as `log_forward(tagged) - log_forward(desupervised)`. |
| CRF-BiRNN | `code/crf.py` (`--withBirnn`) | CRF augmented with a bidirectional sigmoid RNN that produces contextual transition/emission potentials at each position, combining learned tag embeddings, word embeddings, and forward/backward hidden states. |

### Key Algorithms

- **Forward Algorithm** — Log-domain forward pass supporting both supervised (complete-data) and unsupervised (marginalized) cases, with numerically-stable `logsumexp` (`safe_inf` handling of structural zeros).
- **Viterbi Decoding** — Vectorized decoding with backpointer reconstruction (loop-based variant also available in `code/hmm_HQ.py`).
- **Posterior Decoding** — Marginal decoding via autograd (in `code/hmm_HQ.py`), extracting posterior tag probabilities from the forward algorithm's computation graph.
- **Semi-supervised Training** — Unsupervised forward pass marginalizes over tags, enabling EM-like training on raw text combined with supervised data.

## Project Structure

```
├── code/                        # Source code
│   ├── tag.py                   # Command-line training & evaluation entry point
│   ├── hmm.py                   # HMM implementation (vectorized Viterbi)
│   ├── hmm_HQ.py                # HMM variant (loop-based Viterbi + posterior decoding)
│   ├── crf.py                   # CRF and CRF-BiRNN implementation
│   ├── corpus.py                # TaggedCorpus reader and sentence utilities
│   ├── lexicon.py               # Word-feature matrix builder (one-hot, embeddings, affixes, log-counts)
│   ├── lexicon_HQ.py            # Lexicon variant (baseline, no affix features)
│   ├── integerize.py            # Bidirectional string ↔ integer mapping utility
│   ├── logsumexp_safe.py        # Numerically-safe logsumexp/logaddexp with safe_inf
│   ├── eval.py                  # Evaluation: cross-entropy, error rate, output writer
│   ├── test_ic.py               # Ice-cream HMM experiment
│   ├── test_ic_crf.py           # Ice-cream CRF experiment
│   ├── test_en.py               # English HMM experiment
│   ├── test_en_check.py         # English HMM evaluation of saved model
│   ├── nlp-class.yml            # Conda environment specification
│   └── environment.yml          # Conda environment (autograder-compatible)
├── data/                        # Datasets
│   ├── ic*                      # Ice-cream toy data
│   ├── en*                      # English data (supervised, dev, raw)
│   ├── cz*                      # Czech data (supervised, dev, raw)
│   └── my_hmm.pkl               # Saved HMM model
├── models/                      # Pre-trained models
│   ├── en_hmm.pkl               # English HMM (supervised)
│   ├── en_hmm_raw.pkl           # English HMM (semi-supervised)
│   └── en_hmm_awesome.pkl       # English HMM (with log-counts features)
└── docs/                        # Reference materials
    ├── hmm.xls                  # Forward-backward spreadsheet
    ├── hmm-viterbi.xls          # Viterbi spreadsheet
    └── hw-tag.pdf               # Detailed technical handout
```

## Setup

Create and activate the conda environment:

```bash
cd code
conda env create -f nlp-class.yml
conda activate nlp-class
```

**Dependencies:** Python 3.9, PyTorch 1.9, NumPy, SciPy, tqdm, more-itertools, torchtyping

## Usage

### Command-Line Tagger (`tag.py`)

```bash
cd code
python3 tag.py <eval_file> --model <model_file> [options]
```

**Key options:**

| Option | Description |
|--------|-------------|
| `--train <files...>` | Training data files (supervised and/or raw); omit to just evaluate a saved model |
| `--crf` | Use CRF instead of HMM |
| `--withBirnn` | Use CRF with BiRNN features (implies `--crf`) |
| `--awesome` | Enable log-counts lexicon features for HMM |
| `--unigram` | Use unigram (zeroth-order) transition model |
| `--lexicon <file>` | Pretrained word embeddings file |
| `--reg <float>` | L2 regularization strength (default: 1.0) |
| `--lr <float>` | Learning rate (default: 1e-4) |
| `--tolerance <float>` | Early-stopping tolerance on dev loss (default: 0.01) |
| `--train_batch_size <int>` | Minibatch size for training (default: 30) |
| `--eval_batch_size <int>` | Minibatch size for evaluation (default: 2000) |
| `--max_iters <int>` | Maximum training iterations |
| `--save_step <int>` | Save checkpoint every N iterations |
| `--save_path <path>` | Model save path |
| `--output_file <path>` | Output file for Viterbi taggings |
| `-v` / `-q` | Verbose / Quiet mode |

### Examples

**Supervised HMM:**
```bash
python3 tag.py endev --model en_hmm.pkl --train ensup
```

**Semi-supervised HMM (add unlabeled data):**
```bash
python3 tag.py endev --model en_hmm_raw.pkl --train ensup enraw
```

**HMM with improved features:**
```bash
python3 tag.py endev --model en_hmm_awesome.pkl --train ensup --awesome --lexicon words-50.txt
```

**CRF:**
```bash
python3 tag.py endev --model en_crf.pkl --train ensup --crf --lexicon words-50.txt
```

**CRF with BiRNN features:**
```bash
python3 tag.py endev --model en_crf_birnn.pkl --train ensup --crf --withBirnn --lexicon words-50.txt
```

**Evaluate a saved model (no training):**
```bash
python3 tag.py endev --model en_hmm.pkl
```

Output Viterbi taggings are written to `<eval_file>.output` (e.g., `endev.output`).

### Experiment Scripts

```bash
python3 test_ic.py          # Ice-cream HMM full lifecycle
python3 test_ic_crf.py      # Ice-cream CRF experiment
python3 test_en.py          # English HMM training & evaluation
python3 test_en_check.py    # Evaluate a pre-trained English HMM
```

### Interactive Use

```python
from corpus import *
from hmm import *
from lexicon import *

corpus = TaggedCorpus(Path("icsup"))
model = HiddenMarkovModel(corpus, unigram=False)
# ... train and evaluate interactively
```

## Datasets

All datasets are in the `data/` directory. Format: one sentence per line, tokens as `word/tag` (supervised) or bare `word` (raw).

### Ice-Cream (Toy)

A classic toy dataset where "words" are ice-cream consumption counts (1/2/3) and tags are weather states (H=Hot, C=Cold).

| File | Description |
|------|-------------|
| `icsup` | 4 supervised sentences |
| `icraw` | 1 unsupervised sequence (33 observations) |
| `icdev` | 1 labeled dev sentence |

### English (WSJ-style)

Penn Treebank-style data with ~25 tag types (N, V, D, J, P, etc.).

| File | Description |
|------|-------------|
| `ensup` | 4,051 supervised sentences |
| `ensup25k` | ~25k token subset (1,021 sentences) |
| `ensup10k` | ~10k token subset (401 sentences) |
| `ensup4k` | ~4k token subset (160 sentences) |
| `enraw` | 4,013 unsupervised sentences (raw text) |
| `endev` | 996 dev sentences |
| `endev_tiny` | 10 sentences (quick sanity check) |

### Czech

| File | Description |
|------|-------------|
| `czsup` | 8,439 supervised sentences |
| `czdev` | 2,148 dev sentences |
| `czraw` | 8,030 unsupervised sentences |

## Lexicon Features

The `lexicon.py` module builds a word-feature matrix by concatenating feature blocks:

| Feature | Description |
|---------|-------------|
| One-hot | Identity matrix (one column per word) |
| Embeddings | Pretrained word vectors from file |
| Log-counts | `log(1 + count(tag, word))` features from supervised data (enabled by `--awesome`) |
| Affixes | Binary features for common English prefixes/suffixes |

## Evaluation

The evaluation pipeline (`eval.py`) reports:

- **Cross-entropy** per token (in nats) and perplexity
- **Error rate** broken into categories:
  - **ALL** — all tokens
  - **KNOWN** — tokens seen in supervised training
  - **SEEN** — tokens in vocabulary but not supervised
  - **NOVEL** — out-of-vocabulary tokens

## Pre-trained Models

Located in `models/`:

| Model | Training |
|-------|----------|
| `en_hmm.pkl` | Supervised on `ensup` |
| `en_hmm_raw.pkl` | Semi-supervised on `ensup` + `enraw` |
| `en_hmm_awesome.pkl` | Supervised on `ensup` with log-counts features |

## Numerical Stability

The `logsumexp_safe.py` module patches PyTorch's `logsumexp`/`logaddexp` with `safe_inf=True` support, handling the `(-inf ⊕ -inf)` case that arises in forward algorithms with structural zeros (e.g., impossible BOS→BOS transitions). This prevents NaN gradients during backpropagation.
