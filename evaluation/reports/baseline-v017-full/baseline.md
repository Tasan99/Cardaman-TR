# Frozen baseline — tr-aml-v1 v1.0.0

Frozen 2026-09-24T12:14:19.897111+00:00 from `C:\Users\tasan\AppData\Local\Temp\claude\C--Users-tasan-OneDrive-Masa-st--Cardaman\e6854f9b-1c16-4949-9e6b-0c96dd0541d8\scratchpad\oldrun\20260923-181124-ollama-4b-8b-ollama` · Cardaman 0.17.0 · manifest `77d976b9f0f3` · 30 cases

**Mode:** provider `ollama`, retrieval `hybrid`; extraction `qwen3:4b`, judge `qwen3:8b` (thinking `on`), overrides `{}`.

| Metric | Value |
|---|---|
| applicability_accuracy | 0.7609 |
| applies_precision | 0.9091 |
| applies_recall | 0.9434 |
| applies_f1 | 0.9259 |
| does_not_apply_precision | 0.85 |
| does_not_apply_recall | 0.5152 |
| does_not_apply_f1 | 0.6415 |
| unknown_rate | 0.1098 |
| unknown_rate_applicability | — |
| coverage_macro_f1 | 0.6608 |
| coverage_covers_text_precision | 1.0 |
| coverage_covers_text_recall | 0.7273 |
| coverage_covers_text_f1 | 0.8421 |
| coverage_covers_text_support | 22 |
| coverage_partial_precision | 0.1667 |
| coverage_partial_recall | 0.3333 |
| coverage_partial_f1 | 0.2222 |
| coverage_partial_support | 3 |
| coverage_conflict_precision | 0.8889 |
| coverage_conflict_recall | 0.8 |
| coverage_conflict_f1 | 0.8421 |
| coverage_conflict_support | 10 |
| coverage_no_evidence_precision | 0.6364 |
| coverage_no_evidence_recall | 0.875 |
| coverage_no_evidence_f1 | 0.7369 |
| coverage_no_evidence_support | 8 |
| coverage_unknown_precision | 0.0 |
| coverage_unknown_recall | — |
| coverage_unknown_f1 | — |
| coverage_unknown_support | 0 |
| conflict_precision | 0.8889 |
| conflict_recall | 0.8 |
| conflict_fpr | 0.0303 |
| entity_gate_false_inclusion_rate | 0.0 |
| entity_gate_false_exclusion_rate | 0.0 |
| entity_gate_false_inclusions | 0 |
| entity_gate_false_exclusions | 0 |
| recall_at_3 | 0.9722 |
| recall_at_5 | 1.0 |
| recall_at_10 | 1.0 |
| mrr | 0.9375 |
| evidence_hit_rate | 0.8611 |
| proposal_success_rate | 1.0 |
| runtime_seconds | 11072.0 |
| seconds_per_obligation | 119.054 |
| model_seconds | 10981.5 |
| model_seconds_per_obligation | 118.0806 |
| llm_calls | 1843 |
| calls_per_obligation | 19.8172 |
| live_calls | 705 |
| prompt_tokens | 650508 |
| output_tokens | 224226 |
| cache_hit_ratio | 0.6175 |
| model_failures | 1 |
| model_failure_rate | 0.0005 |
| timeouts | 0 |
| malformed_responses | 20 |
| context_overflows | — |
| retries | — |
| cases | 30 |
| obligations | 93 |
| scored_applicability_rows | 92 |

Every file in this directory is listed with its sha256 in FROZEN.json and was made read-only; the regression gate fails when one of them changes.
