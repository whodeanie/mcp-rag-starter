# Evaluation Harness

This directory contains the evaluation harness for the RAG system.

## Files

- `eval_set.json`: 20 hand-crafted question-answer pairs grounded in public domain documents
- `run_eval.py`: Evaluation script that measures retrieval quality

## Metrics

The evaluation script computes:

- Recall@1, Recall@3, Recall@5, Recall@10: Percentage of queries where relevant documents appear in top K results
- MRR (Mean Reciprocal Rank): Average of 1 over the rank position of the first relevant result

## Running Evaluation

To run evaluation on indexed documents:

```bash
python -m evaluation.run_eval
```

With custom paths:

```bash
python -m evaluation.run_eval config.yaml evaluation/eval_set.json examples/corpus.json
```

## Dataset

The eval set consists of 20 questions designed to test different aspects of RAG:

- Questions with answers directly in the corpus (should achieve high recall)
- Paraphrased questions (test semantic understanding)
- Questions with no answer in corpus (test false positive rate)

## Notes

The evaluation uses simple keyword matching against retrieved document content. For production systems, consider using more sophisticated evaluation techniques such as semantic similarity or LLM-based assessments.
