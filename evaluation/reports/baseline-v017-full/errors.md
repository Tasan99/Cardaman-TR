# Error taxonomy — `C:\Users\tasan\AppData\Local\Temp\claude\C--Users-tasan-OneDrive-Masa-st--Cardaman\e6854f9b-1c16-4949-9e6b-0c96dd0541d8\scratchpad\oldrun\20260923-181124-ollama-4b-8b-ollama`

30 cases (30 packets), scored against ..\evaluation\datasets\tr-aml-v1.json. 23 applicability errors, 11 coverage/conflict errors, 0 failed cases, 1 unexpected candidates (not classified).

Rules: the docstring of `regchain.evaluation.taxonomy`; the first rule that holds names the error, the others are listed under *also*.

| Taxonomy | Applicability | Coverage / conflict |
|---|---|---|
| SUBJECT_ENTITY_MISMATCH | 2 | 0 |
| GLOBAL_SCOPE_MISMATCH | 14 | 0 |
| PROFILE_TOO_AMBIGUOUS | 3 | 0 |
| RULE_MODEL_DISAGREEMENT | 3 | 0 |
| MODEL_HALLUCINATION | 0 | 2 |
| PARSER_EXTRACTION_ERROR | 1 | 0 |
| LABEL_AMBIGUITY | 0 | 1 |
| OTHER | 0 | 8 |
| **total** | 23 | 11 |

## Applicability errors

### SUBJECT_ENTITY_MISMATCH (2)

| Where | Exp → actual | Rationale | Also |
|---|---|---|---|
| C28 md.42(1) | DOES_NOT_APPLY → UNKNOWN | The duty is addressed to an authority or another actor (subject "Kendisinden açıklama talep edilen yolcu"), not the company; predicted UNKNOWN. |  |
| C28 md.42(3) | DOES_NOT_APPLY → UNKNOWN | The duty is addressed to an authority or another actor (subject "Tutanaklar"), not the company; predicted UNKNOWN. |  |

### GLOBAL_SCOPE_MISMATCH (14)

| Where | Exp → actual | Rationale | Also |
|---|---|---|---|
| C04 md.5(2) | DOES_NOT_APPLY → UNKNOWN | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet UNKNOWN. | MODEL_HALLUCINATION |
| C04 md.5(3) | DOES_NOT_APPLY → UNKNOWN | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet UNKNOWN. | MODEL_HALLUCINATION |
| C04 md.28(2) | DOES_NOT_APPLY → UNKNOWN | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet UNKNOWN. | MODEL_HALLUCINATION |
| C05 md.46(1) | DOES_NOT_APPLY → UNKNOWN | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet UNKNOWN. | MODEL_HALLUCINATION |
| C06 md.5(2) | DOES_NOT_APPLY → APPLIES | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet APPLIES. |  |
| C06 md.5(3) | DOES_NOT_APPLY → APPLIES | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet APPLIES. |  |
| C06 md.8(1) | DOES_NOT_APPLY → APPLIES | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet APPLIES. | PROFILE_TOO_AMBIGUOUS |
| C06 md.8(3) | DOES_NOT_APPLY → APPLIES | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet APPLIES. | PROFILE_TOO_AMBIGUOUS |
| C06 md.8(5) | DOES_NOT_APPLY → APPLIES | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet APPLIES. | PARENT_CHILD_LEAKAGE, PROFILE_TOO_AMBIGUOUS |
| C07 md.5(2) | DOES_NOT_APPLY → UNKNOWN | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet UNKNOWN. | MODEL_HALLUCINATION |
| C07 md.5(3) | DOES_NOT_APPLY → UNKNOWN | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet UNKNOWN. | MODEL_HALLUCINATION |
| C07 md.28(2) | DOES_NOT_APPLY → UNKNOWN | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet UNKNOWN. | MODEL_HALLUCINATION |
| C24 md.20(1) | DOES_NOT_APPLY → UNKNOWN | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet UNKNOWN. | PROFILE_TOO_AMBIGUOUS, MODEL_HALLUCINATION |
| C24 md.20(2) | DOES_NOT_APPLY → UNKNOWN | Not an obliged party (the label: the company is not in the obliged-party list (md. 4)), yet UNKNOWN. | PROFILE_TOO_AMBIGUOUS, MODEL_HALLUCINATION |

### PROFILE_TOO_AMBIGUOUS (3)

| Where | Exp → actual | Rationale | Also |
|---|---|---|---|
| C25 md.8(1) | UNKNOWN → DOES_NOT_APPLY | The profile does not settle the clause (the clause gate read UNDETERMINED); predicted DOES_NOT_APPLY, labelled UNKNOWN. |  |
| C25 md.8(3) | UNKNOWN → DOES_NOT_APPLY | The profile does not settle the clause (the clause gate read UNDETERMINED); predicted DOES_NOT_APPLY, labelled UNKNOWN. |  |
| C25 md.8(5) | UNKNOWN → DOES_NOT_APPLY | The profile does not settle the clause (the clause gate read UNDETERMINED); predicted DOES_NOT_APPLY, labelled UNKNOWN. |  |

### RULE_MODEL_DISAGREEMENT (3)

| Where | Exp → actual | Rationale | Also |
|---|---|---|---|
| C03 md.8(1) | APPLIES → UNKNOWN | The model said DOES_NOT_APPLY against its own verified basis (downgraded to UNKNOWN); entity gate MATCH. | MODEL_HALLUCINATION |
| C03 md.8(3) | APPLIES → UNKNOWN | The model said DOES_NOT_APPLY against its own verified basis (downgraded to UNKNOWN); entity gate MATCH. | MODEL_HALLUCINATION |
| C03 md.8(5) | APPLIES → UNKNOWN | The model said DOES_NOT_APPLY against its own verified basis (downgraded to UNKNOWN); entity gate MATCH. | MODEL_HALLUCINATION |

### PARSER_EXTRACTION_ERROR (1)

| Where | Exp → actual | Rationale | Also |
|---|---|---|---|
| C26 md.27(3) | APPLIES → NOT_EXTRACTED | No extracted duty matched this required expectation (article, clause and action keywords). |  |

## Coverage and conflict errors

### MODEL_HALLUCINATION (2)

| Where | Exp → actual | Rationale | Also |
|---|---|---|---|
| C14 md.26(2) | COVERS_TEXT → CONFLICT, conflict False → True | The judge claimed a conflict the label does not have. | OTHER |
| C15 md.5(2) | NO_EVIDENCE → PARTIAL | The judge claimed PARTIAL where the label expects NO_EVIDENCE. |  |

### LABEL_AMBIGUITY (1)

| Where | Exp → actual | Rationale | Also |
|---|---|---|---|
| C12 md.5(2) | PARTIAL → NO_EVIDENCE | Disputed label: aml-partial.md: 5(2) is labelled PARTIAL and 5(3) is unasserted; the line between the two is disputed. | OTHER |

### OTHER (8)

| Where | Exp → actual | Rationale | Also |
|---|---|---|---|
| C02 md.6(1) | COVERS_TEXT → UNKNOWN | A passage judgement failed (ProviderFailure). |  |
| C08 md.6(1) | COVERS_TEXT → PARTIAL | The expected passage was retrieved at rank 1 and judged PARTIAL. |  |
| C09 md.7(1) | COVERS_TEXT → PARTIAL | The expected passage was retrieved at rank 1 and judged PARTIAL. |  |
| C11 md.46(1) | COVERS_TEXT → PARTIAL | The expected passage was retrieved at rank 1 and judged PARTIAL. |  |
| C12 md.46(1) | PARTIAL → NO_EVIDENCE | The expected passage was retrieved at rank 2 and judged NO_EVIDENCE without being cited. |  |
| C13 md.5(2) | CONFLICT → NO_EVIDENCE, conflict True → False | The expected passage was retrieved at rank 4 and judged NO_EVIDENCE without being cited. |  |
| C23 md.20(2) | COVERS_TEXT → PARTIAL | The expected passage was retrieved at rank 1 and judged PARTIAL. |  |
| C30 md.5(2) | CONFLICT → NO_EVIDENCE, conflict True → False | The expected passage was retrieved at rank 1 and judged NO_EVIDENCE. |  |
