# Evaluation run — tr-aml-v1 v1.0.0

Created 2026-09-23T18:11:24.673219+00:00 · Cardaman 0.17.0 · source tree `5c2af1f123cc` · manifest `77d976b9f0f3`

**Mode:** provider `ollama`, retrieval `hybrid`, reranker `off`; models: extraction `qwen3:4b`, judge `qwen3:8b` (thinking `on`), embedding `bge-m3:latest`, reranker `off`.

## Headline

| Metric | Value |
|---|---|
| Cases / obligations scored | 30 / 92 |
| Applicability accuracy | 76.1% |
| Applicability P / R / F1 (APPLIES) | 90.9% / 94.3% / 92.6% |
| Applicability macro F1 | 60.9% |
| Coverage macro P / R / F1 | 67.3% / 68.4% / 66.1% |
| Conflict P / R / FPR | 88.9% / 80.0% / 3.0% |
| Entity gate false exclusion / inclusion | 0.0% / 0.0% |
| Retrieval R@3 / R@5 / R@10 / MRR | 97.2% / 100.0% / 100.0% / 0.9375 |
| Evidence hit rate | 86.1% |
| Extraction recall / precision | 98.9% / 98.9% |
| Proposal success / draft schema validity | 100.0% / 100.0% |
| UNKNOWN rate / model failure / timeout / malformed | 11.0% / 0.1% / 0.0% / 1.1% |
| Wall clock / model seconds / LLM calls | 11072.0 s / 10981.5 s / 1843 |
| Tokens (prompt / output) · cache hit ratio | 650508 / 224226 · 61.8% |
| Seconds per obligation | 119.054 |

## Applicability

| expected \ predicted | APPLIES | DOES_NOT_APPLY | UNKNOWN | OTHER |
|---|---|---|---|---|
| APPLIES | 50 | 0 | 3 | 0 |
| DOES_NOT_APPLY | 5 | 17 | 11 | 0 |
| UNKNOWN | 0 | 3 | 3 | 0 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| APPLIES | 90.9% | 94.3% | 92.6% | 53 |
| DOES_NOT_APPLY | 85.0% | 51.5% | 64.1% | 33 |
| UNKNOWN | 17.6% | 50.0% | 26.1% | 6 |

## Policy coverage

| expected \ predicted | COVERS_TEXT | PARTIAL | CONFLICT | NO_EVIDENCE | UNKNOWN | OTHER |
|---|---|---|---|---|---|---|
| COVERS_TEXT | 16 | 4 | 1 | 0 | 1 | 0 |
| PARTIAL | 0 | 1 | 0 | 2 | 0 | 0 |
| CONFLICT | 0 | 0 | 8 | 2 | 0 | 0 |
| NO_EVIDENCE | 0 | 1 | 0 | 7 | 0 | 0 |
| UNKNOWN | 0 | 0 | 0 | 0 | 0 | 0 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| COVERS_TEXT | 100.0% | 72.7% | 84.2% | 22 |
| PARTIAL | 16.7% | 33.3% | 22.2% | 3 |
| CONFLICT | 88.9% | 80.0% | 84.2% | 10 |
| NO_EVIDENCE | 63.6% | 87.5% | 73.7% | 8 |
| UNKNOWN | 0.0% | — | — | 0 |

Not-assessed expected / correct: 13 / 13

## Conflict

| TP | FP | FN | TN | Precision | Recall | FPR |
|---|---|---|---|---|---|---|
| 8 | 1 | 2 | 32 | 88.9% | 80.0% | 3.0% |

## Entity gate

| Rows | Expected excluded | False exclusions | False inclusions |
|---|---|---|---|
| 26 | 13 | 0 | 0 |

## Retrieval

| n | R@3 | R@5 | R@10 | MRR | Evidence hit |
|---|---|---|---|---|---|
| 36 | 97.2% | 100.0% | 100.0% | 0.9375 | 86.1% |

## Proposals and reliability

| Needed | Produced | Drafts attempted | Schema valid | UNKNOWN | Failures | Timeouts | Malformed |
|---|---|---|---|---|---|---|---|
| 39 | 39 | 39 | 100.0% | 19 | 1 | 0 | 20 |

## Performance

| Task | Calls |
|---|---|
| judge.supports | 705 |
| judge.contradicts | 607 |
| extraction | 152 |
| extraction.review | 93 |
| enrich | 93 |
| extraction.repair | 74 |
| judge.applicability | 60 |
| draft | 40 |
| judge.conflict_confirm | 19 |

Slowest cases: C03 (1500.4 s), C02 (1352.2 s), C11 (903.4 s), C04 (796.6 s), C28 (766.5 s)

## Cases

| Case | Obligation | Applicability (exp → pred) | Coverage (exp → pred) | Conflict | Evidence rank | s |
|---|---|---|---|---|---|---|
| C01 | md.4(2) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 76 |
| C01 | md.8(1) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 76 |
| C01 | md.8(2) | DOES_NOT_APPLY | not extracted (optional) |  |  |  |
| C01 | md.8(3) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 76 |
| C01 | md.8(5) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 76 |
| C02 | md.5(1) | APPLIES | not extracted (optional) |  |  |  |
| C02 | md.5(2) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 1352 |
| C02 | md.5(3) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 1352 |
| C02 | md.6(1) | APPLIES → APPLIES ✓ | COVERS_TEXT → UNKNOWN ✗ | False → False ✓ | 1 | 1352 |
| C02 | md.7(1) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 1352 |
| C02 | md.7(4) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 1352 |
| C03 | md.8(1) | APPLIES → UNKNOWN ✗ | NO_EVIDENCE → NO_EVIDENCE ✓ | False → False ✓ |  | 1500 |
| C03 | md.8(2) | APPLIES | not extracted (optional) |  |  |  |
| C03 | md.8(3) | APPLIES → UNKNOWN ✗ | — |  |  | 1500 |
| C03 | md.8(5) | APPLIES → UNKNOWN ✗ | — |  |  | 1500 |
| C03 | md.9(1) | APPLIES → APPLIES ✓ | NO_EVIDENCE → NO_EVIDENCE ✓ | False → False ✓ |  | 1500 |
| C03 | md.9(2) | APPLIES → APPLIES ✓ | — |  |  | 1500 |
| C04 | md.5(1) | DOES_NOT_APPLY | not extracted (optional) |  |  |  |
| C04 | md.5(2) | DOES_NOT_APPLY → UNKNOWN ✗ | — |  |  | 797 |
| C04 | md.5(3) | DOES_NOT_APPLY → UNKNOWN ✗ | — |  |  | 797 |
| C04 | md.28(2) | DOES_NOT_APPLY → UNKNOWN ✗ | — |  |  | 797 |
| C04 | md.46(1) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | — |  |  | 797 |
| C05 | md.5(1) | DOES_NOT_APPLY | not extracted (optional) |  |  |  |
| C05 | md.5(2) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | — |  |  | 192 |
| C05 | md.5(3) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | — |  |  | 192 |
| C05 | md.46(1) | DOES_NOT_APPLY → UNKNOWN ✗ | — |  |  | 192 |
| C06 | md.5(1) | DOES_NOT_APPLY | not extracted (optional) |  |  |  |
| C06 | md.5(2) | DOES_NOT_APPLY → APPLIES ✗ | — |  |  | 188 |
| C06 | md.5(3) | DOES_NOT_APPLY → APPLIES ✗ | — |  |  | 188 |
| C06 | md.8(1) | DOES_NOT_APPLY → APPLIES ✗ | — |  |  | 188 |
| C06 | md.8(2) | DOES_NOT_APPLY | not extracted (optional) |  |  |  |
| C06 | md.8(3) | DOES_NOT_APPLY → APPLIES ✗ | — |  |  | 188 |
| C06 | md.8(5) | DOES_NOT_APPLY → APPLIES ✗ | — |  |  | 188 |
| C07 | md.5(1) | DOES_NOT_APPLY | not extracted (optional) |  |  |  |
| C07 | md.5(2) | DOES_NOT_APPLY → UNKNOWN ✗ | — |  |  | 458 |
| C07 | md.5(3) | DOES_NOT_APPLY → UNKNOWN ✗ | — |  |  | 458 |
| C07 | md.28(2) | DOES_NOT_APPLY → UNKNOWN ✗ | — |  |  | 458 |
| C07 | md.46(1) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | — |  |  | 458 |
| C08 | md.6(1) | APPLIES → APPLIES ✓ | COVERS_TEXT → PARTIAL ✗ | False → False ✓ | 1 | 122 |
| C08 | md.7(1) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 122 |
| C08 | md.7(4) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 122 |
| C09 | md.6(1) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 582 |
| C09 | md.7(1) | APPLIES → APPLIES ✓ | COVERS_TEXT → PARTIAL ✗ | False → False ✓ | 1 | 582 |
| C09 | md.7(4) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 582 |
| C10 | md.5(1) | UNKNOWN | not extracted (optional) |  |  |  |
| C10 | md.5(2) | UNKNOWN → UNKNOWN ✓ | — |  |  | 155 |
| C10 | md.5(3) | UNKNOWN → UNKNOWN ✓ | — |  |  | 155 |
| C10 | md.28(2) | UNKNOWN → UNKNOWN ✓ | — |  |  | 155 |
| C11 | md.5(1) | APPLIES | not extracted (optional) |  |  |  |
| C11 | md.5(2) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 903 |
| C11 | md.5(3) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 2 | 903 |
| C11 | md.28(2) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 903 |
| C11 | md.46(1) | APPLIES → APPLIES ✓ | COVERS_TEXT → PARTIAL ✗ | False → False ✓ | 1 | 903 |
| C12 | md.5(1) | APPLIES | not extracted (optional) |  |  |  |
| C12 | md.5(2) | APPLIES → APPLIES ✓ | PARTIAL → NO_EVIDENCE ✗ | False → False ✓ | 2 | 499 |
| C12 | md.5(3) | APPLIES → APPLIES ✓ | — |  |  | 499 |
| C12 | md.28(2) | APPLIES → APPLIES ✓ | PARTIAL → PARTIAL ✓ | False → False ✓ | 1 | 499 |
| C12 | md.46(1) | APPLIES → APPLIES ✓ | PARTIAL → NO_EVIDENCE ✗ | False → False ✓ | 2 | 499 |
| C13 | md.5(1) | APPLIES | not extracted (optional) |  |  |  |
| C13 | md.5(2) | APPLIES → APPLIES ✓ | CONFLICT → NO_EVIDENCE ✗ | True → False ✗ | 4 | 616 |
| C13 | md.5(3) | APPLIES → APPLIES ✓ | — |  |  | 616 |
| C13 | md.28(2) | APPLIES → APPLIES ✓ | CONFLICT → CONFLICT ✓ | True → True ✓ | 1 | 616 |
| C13 | md.46(1) | APPLIES → APPLIES ✓ | CONFLICT → CONFLICT ✓ | True → True ✓ | 1 | 616 |
| C14 | md.26(2) | APPLIES → APPLIES ✓ | COVERS_TEXT → CONFLICT ✗ | False → True ✗ | 1 | 304 |
| C14 | md.28(2) | APPLIES → APPLIES ✓ | CONFLICT → CONFLICT ✓ | True → True ✓ | 1 | 304 |
| C15 | md.5(1) | APPLIES | not extracted (optional) |  |  |  |
| C15 | md.5(2) | APPLIES → APPLIES ✓ | NO_EVIDENCE → PARTIAL ✗ | False → False ✓ |  | 277 |
| C15 | md.5(3) | APPLIES → APPLIES ✓ | NO_EVIDENCE → NO_EVIDENCE ✓ | False → False ✓ |  | 277 |
| C15 | md.46(1) | APPLIES → APPLIES ✓ | NO_EVIDENCE → NO_EVIDENCE ✓ | False → False ✓ |  | 277 |
| C16 | md.5(1) | APPLIES | not extracted (optional) |  |  |  |
| C16 | md.5(2) | APPLIES → APPLIES ✓ | NO_EVIDENCE → NO_EVIDENCE ✓ | False → False ✓ |  | 357 |
| C16 | md.5(3) | APPLIES → APPLIES ✓ | — |  |  | 357 |
| C16 | md.28(2) | APPLIES → APPLIES ✓ | CONFLICT → CONFLICT ✓ | True → True ✓ | 1 | 357 |
| C16 | md.46(1) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 357 |
| C17 | md.5(1) | APPLIES | not extracted (optional) |  |  |  |
| C17 | md.5(2) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 73 |
| C17 | md.5(3) | APPLIES → APPLIES ✓ | — |  |  | 73 |
| C18 | md.5(1) | APPLIES | not extracted (optional) |  |  |  |
| C18 | md.5(2) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 396 |
| C18 | md.5(3) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 396 |
| C18 | md.46(1) | APPLIES → APPLIES ✓ | CONFLICT → CONFLICT ✓ | True → True ✓ | 1 | 396 |
| C19 | md.46(1) | APPLIES → APPLIES ✓ | NO_EVIDENCE → NO_EVIDENCE ✓ | False → False ✓ | 1 | 82 |
| C20 | md.5(1) | APPLIES | not extracted (optional) |  |  |  |
| C20 | md.5(2) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 160 |
| C20 | md.5(3) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 160 |
| C20 | md.28(2) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 160 |
| C20 | md.46(1) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 160 |
| C21 | md.4(2) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 1 |
| C22 | md.4(2) | APPLIES → APPLIES ✓ | — |  |  | 244 |
| C23 | md.20(1) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 590 |
| C23 | md.20(2) | APPLIES → APPLIES ✓ | COVERS_TEXT → PARTIAL ✗ | False → False ✓ | 1 | 590 |
| C24 | md.20(1) | DOES_NOT_APPLY → UNKNOWN ✗ | — |  |  | 69 |
| C24 | md.20(2) | DOES_NOT_APPLY → UNKNOWN ✗ | — |  |  | 69 |
| C25 | md.8(1) | UNKNOWN → DOES_NOT_APPLY ✗ | — |  |  | 64 |
| C25 | md.8(2) | UNKNOWN | not extracted (optional) |  |  |  |
| C25 | md.8(3) | UNKNOWN → DOES_NOT_APPLY ✗ | — |  |  | 64 |
| C25 | md.8(5) | UNKNOWN → DOES_NOT_APPLY ✗ | — |  |  | 64 |
| C26 | md.27(3) | APPLIES | **not extracted** |  |  |  |
| C26 | md.28(2) | APPLIES → APPLIES ✓ | COVERS_TEXT → COVERS_TEXT ✓ | False → False ✓ | 1 | 17 |
| C28 | md.42(1) | DOES_NOT_APPLY → UNKNOWN ✗ | — |  |  | 767 |
| C28 | md.42(2) | DOES_NOT_APPLY | not extracted (optional) |  |  |  |
| C28 | md.42(3) | DOES_NOT_APPLY → UNKNOWN ✗ | — |  |  | 767 |
| C29 | md.5(1) | APPLIES | not extracted (optional) |  |  |  |
| C29 | md.5(2) | APPLIES → APPLIES ✓ | NO_EVIDENCE → NO_EVIDENCE ✓ | False → False ✓ |  | 128 |
| C29 | md.5(3) | APPLIES → APPLIES ✓ | — |  |  | 128 |
| C29 | md.8(1) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 128 |
| C29 | md.8(2) | DOES_NOT_APPLY | not extracted (optional) |  |  |  |
| C29 | md.8(3) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 128 |
| C29 | md.8(5) | DOES_NOT_APPLY → DOES_NOT_APPLY ✓ | NOT_ASSESSED → NOT_ASSESSED ✓ |  |  | 128 |
| C29 | md.28(2) | APPLIES → APPLIES ✓ | CONFLICT → CONFLICT ✓ | True → True ✓ | 1 | 128 |
| C30 | md.5(1) | APPLIES | not extracted (optional) |  |  |  |
| C30 | md.5(2) | APPLIES → APPLIES ✓ | CONFLICT → NO_EVIDENCE ✗ | True → False ✗ | 1 | 102 |
| C30 | md.5(3) | APPLIES → APPLIES ✓ | — |  |  | 102 |
| C30 | md.28(2) | APPLIES → APPLIES ✓ | CONFLICT → CONFLICT ✓ | True → True ✓ | 1 | 102 |
| C30 | md.46(1) | APPLIES → APPLIES ✓ | CONFLICT → CONFLICT ✓ | True → True ✓ | 1 | 102 |

## Known false positives and false negatives

- **applicability_false_negative** (3): C03 md.8(1) [APPLIES → UNKNOWN]; C03 md.8(3) [APPLIES → UNKNOWN]; C03 md.8(5) [APPLIES → UNKNOWN]
- **applicability_false_positive** (5): C06 md.5(2) [DOES_NOT_APPLY → APPLIES]; C06 md.5(3) [DOES_NOT_APPLY → APPLIES]; C06 md.8(1) [DOES_NOT_APPLY → APPLIES]; C06 md.8(3) [DOES_NOT_APPLY → APPLIES]; C06 md.8(5) [DOES_NOT_APPLY → APPLIES]
- **applicability_mismatch** (14): C04 md.5(2) [DOES_NOT_APPLY → UNKNOWN]; C04 md.5(3) [DOES_NOT_APPLY → UNKNOWN]; C04 md.28(2) [DOES_NOT_APPLY → UNKNOWN]; C05 md.46(1) [DOES_NOT_APPLY → UNKNOWN]; C07 md.5(2) [DOES_NOT_APPLY → UNKNOWN]; C07 md.5(3) [DOES_NOT_APPLY → UNKNOWN]; C07 md.28(2) [DOES_NOT_APPLY → UNKNOWN]; C24 md.20(1) [DOES_NOT_APPLY → UNKNOWN]; C24 md.20(2) [DOES_NOT_APPLY → UNKNOWN]; C25 md.8(1) [UNKNOWN → DOES_NOT_APPLY]; C25 md.8(3) [UNKNOWN → DOES_NOT_APPLY]; C25 md.8(5) [UNKNOWN → DOES_NOT_APPLY]; C28 md.42(1) [DOES_NOT_APPLY → UNKNOWN]; C28 md.42(3) [DOES_NOT_APPLY → UNKNOWN]
- **conflict_false_negative** (2): C13 md.5(2); C30 md.5(2)
- **conflict_false_positive** (1): C14 md.26(2)
- **coverage_mismatch** (11): C02 md.6(1) [COVERS_TEXT → UNKNOWN]; C08 md.6(1) [COVERS_TEXT → PARTIAL]; C09 md.7(1) [COVERS_TEXT → PARTIAL]; C11 md.46(1) [COVERS_TEXT → PARTIAL]; C12 md.5(2) [PARTIAL → NO_EVIDENCE]; C12 md.46(1) [PARTIAL → NO_EVIDENCE]; C13 md.5(2) [CONFLICT → NO_EVIDENCE]; C14 md.26(2) [COVERS_TEXT → CONFLICT]; C15 md.5(2) [NO_EVIDENCE → PARTIAL]; C23 md.20(2) [COVERS_TEXT → PARTIAL]; C30 md.5(2) [CONFLICT → NO_EVIDENCE]
- **extraction_miss** (1): C26 md.27(3)
- **unexpected_candidate** (1): C28 md.42(2)

---
Unit tests prove the code does what its author meant; these numbers measure how often the analysis agrees with the expert labels of this dataset, in this mode, with these models. Neither is the product's accuracy in the wild.
