# Hierarchical Transformer for Entity Extraction

> Status: work in progress. Architecture and training pipeline are public; results and pretrained weights are being finalized.

A context-aware transformer for employer and entity extraction from job descriptions. The model fuses reasoning at three levels, sentence, section, and document, so an entity is labeled using the full structure of the text rather than a single sentence in isolation.

## Problem

Job descriptions are messy, semi-structured text. The same token (a company name, a skill, a title) means different things depending on where it sits in the document. Flat token-level models miss this because they see one window at a time and lose the document-wide context that tells you which organization is the actual employer versus a client, partner, or parent company.

## Approach

Three levels of context, combined:

1. Sentence level. A transformer encoder (DistilBERT backbone) produces contextual token embeddings within each sentence.
2. Section level. Sentences are grouped into sections (COMPANY, SKILLS, RESPONSIBILITIES, and similar) and an attention layer mixes signal across sentences in the same section.
3. Document level. TF-IDF-guided pruning highlights the document-wide terms that matter most and down-weights boilerplate, giving the model a global prior over which spans are likely to be the employer.

The three representations are fused before the entity-extraction head, so each prediction is conditioned on local wording, section role, and document-wide salience at once.

## Architecture

```
Tokens
  -> DistilBERT encoder (sentence-level contextual embeddings)
  -> Section attention (cross-sentence mixing within a section)
  -> TF-IDF document prior (global salience weighting)
  -> Fusion layer
  -> Entity / employer extraction head
```

## Results

Being finalized. This table will hold the evaluation once the current training run completes.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Token baseline (DistilBERT) | TBD | TBD | TBD |
| + section attention | TBD | TBD | TBD |
| + document prior (full model) | TBD | TBD | TBD |

## Repository structure

```
data/        sample inputs and preprocessing scripts
src/         model, training, and evaluation code
notebooks/   exploration and error analysis
results/     metrics and figures
```

## Getting started

```
git clone https://github.com/vsancnaj/hierarchical-transformer-entity-extraction.git
cd hierarchical-transformer-entity-extraction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Roadmap

- [ ] Finish full-model training run and populate the results table
- [ ] Add an error-analysis notebook
- [ ] Release a small demo on Hugging Face Spaces
- [ ] Publish pretrained weights

## License

MIT
