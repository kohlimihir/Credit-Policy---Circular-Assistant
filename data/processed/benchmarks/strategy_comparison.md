# Chunking Strategy Comparison

Retrieval metrics are **document-level**, scored on the 100 answerable gold questions that carry labelled source documents.
Abstention metrics use all 120 questions (20 are deliberately unanswerable).

| Strategy | R@1 | R@3 | R@5 | MRR | Faithful | Abstain P/R | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| fixed_512 | 0.623 | 0.740 | 0.772 | 0.819 | 0.579 | 0.36/0.60 | 638ms | 680ms |
| recursive_overlap | 0.583 | 0.767 | 0.807 | 0.812 | 0.632 | 0.33/0.60 | 646ms | 696ms |
| structural | 0.610 | 0.770 | 0.813 | 0.828 | 0.668 | 0.39/0.60 | 517ms | 596ms |
| structural_norerank | 0.608 | 0.755 | 0.820 | 0.827 | 0.672 | n/a | 51ms | 93ms |
