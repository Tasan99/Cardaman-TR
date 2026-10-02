# Cardaman TR — one core, multiple regulatory packs

Status: DESIGN (no code changed by this document). Date: 2026-09-30.
Scope: architecture analysis, pack interface, migration plan, schemas, source adapters,
evaluation plan, test plan and implementation order for `BEVERAGE_ALCOHOL_TR` and
`BEVERAGE_NON_ALCOHOL_TR` on the shared Cardaman TR core, designed so that
`FINANCIAL_SERVICES_TR`, `FOOD_TR`, `RETAIL_TR`, `ENERGY_TR`, `TELECOM_TR`, `PHARMA_TR`
can be added without rebuilding the core.

---

## 0. Findings from the current codebase

### 0.1 What exists and is reusable as core

| Capability | Where | Domain-generic? |
|---|---|---|
| Legislation fetch (host allowlist), content hashing | `ingestion/fetch.py`, `ingestion/models.py` | yes |
| mevzuat.gov.tr parser (MADDE / fıkra / bent) | `ingestion/mevzuat.py` (wired via `pilot/sources.parse_source`) | yes |
| Paragraph diff, source-vs-parser change attribution | `ingestion/diff.py`, `ingestion/compare.py` | yes (textual, not legal) |
| Append-only versioned store | `ingestion/repository.py` | **no — locked to `regulators.code='FCA'` (line 31)** |
| Unit split + drafting classification (definition/scope/delegation/penalty/exemption) | `extraction/classify.py`, `engine.split_units` | yes |
| Obligation extraction (subject, modality, action, conditions, exceptions, evidence) | `extraction/pipeline.py`, `schema.py`, `contract.py`, `providers.py` | yes, but prompt examples are AML (`providers.py` ~274–279) |
| Span grounding, quote verification, quantities | `extraction/spans.py`, `grounding.py`, `quantities.py`, `structure.py` | yes |
| Cross-reference retrieval | `extraction/retrieval.py` | TR article logic generic; FCA `BOOKS` table is UK-specific |
| Hybrid policy retrieval (bge-m3 + lexical + rerank) | `pilot/semantic.py`, `policies.py`, `rerank.py` | yes |
| Rule-first applicability gate chain | `pilot/applicability.rule_chain` (803–836) | **machinery generic, content AML** |
| Clause entity gate (addressee / required entities) | `pilot/entities.py` | **mixed** |
| Policy comparison / contradiction (v18 and v19 paths) | `pilot/engine.py`, `pilot/conflict.py` | **mixed** (`ACTS`, `AUTHORITY`, `INSTITUTION` lexicons are AML) |
| Counted coverage (`COVERS_TEXT / PARTIAL / CONFLICT / NO_EVIDENCE / UNKNOWN`) | `engine.coverage_of`, `coverage_of_v19` | yes |
| Grounding validators, normative anchor audit, source lineage | `engine.validate_evidence`, `normative_audit.py`, `scope_integrity.py` | yes |
| Decision trace | `pilot/decision_trace.py` | yes |
| Change impact vs previous run, carry-forward | `pilot/impact.py` | mixed (change kinds generic, suggested actions TR/AML text) |
| Structured quantity change | `pilot/financial.py` | **pack-specific impact enum** (`CAPITAL_REQUIREMENT`, `CUSTOMER_DUE_DILIGENCE`) |
| Human review (APPROVE / OVERRIDE / NEEDS_EVIDENCE / REJECT, hash-chained) | `pilot/review.py` | yes |
| Audit log, RBAC, redaction, retention, tenant storage | `platform/*` | yes |
| Local-only ModelRouter (qwen3:4b extract, qwen3:8b judge, bge-m3 embed) | `model_router.py` | yes — unchanged by this design |
| Evaluation harness, P/R/F1, FPR, gate, labels overlay, four-eyes label changes, expert gold schema | `evaluation/*` | largely dataset-agnostic |
| **Pack registry, source authority layers, enterprise profile graph, applicability routing** | `tr/core.py`, `tr/packs.py`, `tr/profile.py`, `tr/routing.py` (milestone 1) | yes — this is the seed of the pack system |

### 0.2 Hard-coded financial / AML assumptions (must move behind the pack interface)

| Location | Encodes |
|---|---|
| `pilot/applicability.py` 65–117 `CATEGORIES`; 119–144 `FINANCIAL`, `DNFBP`, `GENERIC_CATEGORIES`, `DEFINED_TERMS`; 169–171 `REGULATOR` (BDDK, SPK, TCMB, MASAK, SEDDK); 172–192 `FINANCE_WORDS`; 196–200 `NON_OBLIGED`; 853–867 `financial_identity` | obliged-party taxonomy of Kanun 5549 / Tedbirler Yönetmeliği |
| `pilot/applicability.py` 432–458 `obliged_list`, list heading cues ("Yükümlüler") | assumes the addressee list is a "Yükümlüler" article |
| `pilot/entities.py` 32–61 `COUNTERPARTIES`, `OBLIGED`; 66–87 customer families / `SATISFIES`; 91 `CLAUSE_HEAD` (kimlik tespiti) | KYC / counterparty model |
| `pilot/conflict.py` 328–336 `ACTS` (`@report`, `@identify`, `@keep`); 457–484 `INSTITUTION`, `GROUPS`, `AUTHORITY` (MASAK, FIU, PEP) | AML act lexicon and actor roles |
| `pilot/engine.py` 210–212 `DEFAULT_CATEGORIES` (already injectable via `analyze(categories=)`); 279–351 prompt note Başkanlık↔MASAK | AML risk categories and synonymy |
| `pilot/financial.py` 17–24, 52–56 | financial impact types, `yükümlü`/`banka` regex |
| `extraction/providers.py` ~274–279 | few-shot examples from 5549 |
| `ingestion/repository.py` 30–32; `ingestion/sources.py` 12–28; `extraction/retrieval.py` `BOOKS`; `extraction/amendments.py` | FCA-only regulator and source catalogue |
| `evaluation/taxonomy.py` 87–109, 214–217 | disputed-label table keyed to `tr-aml-v1`, "yükümlü listesi" heuristics |

The `Company` schema (`pilot/schema.py` 9–19) is generic by field name; its finance
specificity lives in the tables above, not in the schema. `customer_types` is the only
field whose semantics are KYC-shaped.

### 0.3 Missing abstractions

1. No pack interface in `pilot/`: the AML knowledge is module-level constants read directly by the engine.
2. Applicability is company-level (`Company` flat lists); no legal-entity / facility / product / activity target in the engine (the new `tr/routing.py` has it, the engine does not).
3. No declarative CONDITION or EXCEPTION representation that can be evaluated against profile facts; exceptions are detected by regex (`exemption_check`) and model basis only.
4. No explanation record that ties a decision to source + article + actor + activity + product + condition + exception + profile fact.
5. Coverage stops at policy text; there is no department / SOP / control / evidence model.
6. Change impact is per analysis run and per provision text; it does not reach products, facilities, policies, controls or departments.
7. Evaluation lacks FNR as a field, unsupported-conclusion count, product/facility/actor-matching tasks and an `INSUFFICIENT_SAMPLE` status (expert gold only raises `ValueError` below 100/task).
8. Ingestion repository cannot store non-FCA regulators; Resmî Gazete is a title scan only.

### 0.4 Tests

- 1383 backend tests: 88 pack tests (`test_tr_*.py`) pass; the rest of the suite is the pre-existing baseline.
- 242 tests are broken by artefacts still missing after the recovery (`tedbirler-200713012/snapshot.json` 191, `tr-aml-v1.json` 34, launcher/thresholds/independent/others 17). They are not caused or fixed by this design.
- **Must remain unchanged** (AML behaviour contract): `test_v018_applicability`, `test_v019_universal_duty`, `test_scope_hardening`, `test_phase15*`, `test_turkiye`, `test_v019_t6_*`, `test_v019_t7_*`, `test_conflict_hardening`, `test_v019_extraction*`, `test_v019_entity_taxonomy`, all evaluation tests, platform / model-router / ingestion tests.
- **Require extension**: `test_tr_*` (pack contract, profile, routing, contrast), evaluation tests for new metrics and statuses, a new architecture test (core does not import packs), an AML-equivalence test for the adapter step.

---

## 1. Architecture proposal

```
                         ┌─────────────────────────── CARDAMAN TR CORE ───────────────────────────┐
 sources ─► SourceAdapter ─► ingestion (versioned store, diff) ─► extraction (obligations, spans)  │
                         │        │                                   │                           │
                         │        ▼                                   ▼                           │
                         │  RegulationCatalogue ◄── PackRegistry ──► ScopeResolver ◄── EnterpriseProfile
                         │  (authority layers,        (packs +         (targets × dimensions,      │
                         │   version chains)           modules)         Kleene 3-valued logic)     │
                         │        │                                   │                           │
                         │        ▼                                   ▼                           │
                         │  Grounding / validators ──► ApplicabilityDecision (explained) ──► review│
                         │                                   │                                     │
                         │                                   ▼                                     │
                         │  policy retrieval ► comparison ► ObligationMapping (dept/SOP/control/  │
                         │                                   evidence/gap) ► reporting            │
                         │                                   ▲                                     │
                         │  ChangeImpact (version diff ► obligations ► targets ► mappings ► actions)
                         └────────────────────────────────────────────────────────────────────────┘
                                  ▲ data + narrow hooks only (no engine code in packs)
      ┌────────────────┬──────────┴───────┬───────────────────┬──────────────────────┐
  FINANCIAL_SERVICES_TR  BEVERAGE_ALCOHOL_TR  BEVERAGE_NON_ALCOHOL_TR   FOOD_TR, RETAIL_TR, …
  (current AML knowledge) │                  │
                          └──── includes ────┴──► modules: TR_FOOD_COMMON, TR_CONSUMER, TR_ADVERTISING,
                                                  TR_ENVIRONMENT, TR_PACKAGING_WASTE, TR_OHS,
                                                  TR_EMPLOYMENT, TR_DATA_PROTECTION, TR_COMPETITION, TR_TAX
```

Principles:

1. **One engine.** Every stage in §0.1 stays in core. A pack never contains retrieval, extraction, grounding, contradiction, policy comparison, review or audit code.
2. **Packs are data first.** A pack is a directory of validated JSON (manifest, taxonomy, catalogue, scopes, exceptions, ownership, evaluation cases). Optional Python hooks are pure functions behind a declared interface, loaded through entry points; core never imports a pack by name.
3. **Horizontal law lives in modules, not in sector packs.** KVKK, İş Sağlığı ve Güvenliği, employment, competition, consumer protection, e-commerce, advertising, environment, waste and packaging waste are owned once by a horizontal module and included by every pack that needs them. The same obligation therefore produces comparable decisions across packs (the packaging-waste case in the brief).
4. **Single ownership.** A regulation belongs to exactly one pack or module (already enforced by `tr/packs.Registry`). Packs extend, never fork, a regulation.
5. **Applicability is evaluated per target** (legal entity, facility, product, activity) and rolled up; the group answer is a summary, never the unit of decision.
6. **UNKNOWN is a first-class value** everywhere: facts, conditions, exceptions, decisions, coverage, metrics. Three-valued (Kleene) logic; no default-to-NO.
7. **Authority layers decide what may create an obligation**: BINDING creates; OFFICIAL_GUIDANCE interprets; DECISION_PRECEDENT is second phase. Guidance that conflicts with binding text → binding wins, flag `GUIDANCE_CONFLICT`, human review.
8. **Decision stages are explicit**: `ROUTED` (pack metadata only), `GROUNDED` (extracted provision + quotes verified), `REVIEWED` (human). Only GROUNDED or REVIEWED decisions are "final" and must cite the article.

---

## 2. Pack interface design

### 2.1 Pack contents

```
packs/<PACK_ID>/
  manifest.json          id, kind PACK|MODULE, version, jurisdiction, includes[], source_layers[], status
  taxonomy.json          entity/activity/facility/product classes (hierarchical), attribute definitions,
                         sales channels, departments, regulators, obligation categories
  lexicon.json           Turkish surface forms per class (for clause-level actor/activity/product matching)
  catalogue.json         RegulationMeta entries owned by this pack
  scopes.json            ObligationScope entries (PACK_METADATA or EXTRACTED)
  exceptions.json        ExceptionRule entries (grounded or UNVERIFIED)
  ownership.json         obligation category → default departments
  impact_types.json      pack-declared change-impact types (e.g. LABEL_CONTENT, COMPOSITION_LIMIT)
  evaluation/            dev cases, contrast cases, expert-gold manifest (no invented labels)
  hooks.py (optional)    pure functions registered under the interface below
```

### 2.2 Interface (core side)

```python
class PackManifest(Strict):
    format: Literal['cardaman-pack/1']
    pack_id: str                      # ^[A-Z][A-Z0-9_]+$
    kind: Literal['PACK', 'MODULE']
    version: str                      # semver; decisions record pack_id@version
    jurisdiction: Literal['TR']       # widened when a second jurisdiction exists
    includes: list[str] = []          # MODULE ids only, acyclic
    source_layers: list[SourceClass]  # MVP: BINDING, OFFICIAL_GUIDANCE
    status: Literal['ACTIVE', 'DRAFT', 'RETIRED']
    selection: SelectionRule          # how a profile selects this pack (product classes, exclusive licences, facts)

class PackHooks(Protocol):            # all optional; defaults are data-driven
    def derive_facts(self, profile: EnterpriseProfile) -> dict[str, Fact]: ...
    def actor_lexicon(self) -> ActorLexicon: ...          # clause text → classes (replaces CATEGORIES for this pack)
    def risk_categories(self) -> list[str]: ...           # replaces engine.DEFAULT_CATEGORIES
    def conflict_lexicon(self) -> ConflictLexicon: ...    # replaces conflict.ACTS / AUTHORITY for this pack
    def impact_classifier(self, old: str, new: str) -> list[ImpactType]: ...   # replaces financial.py role

class PackRegistry:                   # generalises tr/packs.Registry
    @classmethod
    def discover(cls) -> 'PackRegistry': ...   # data dirs + entry_points(group='cardaman.packs')
    def pack(self, pack_id) -> Pack: ...
    def closure(self, pack_id) -> list[str]: ...
    def select(self, profile) -> list[PackSelection]: ...
    def scopes_for(self, selections) -> list[RoutedScope]: ...
```

Rules enforced at load (extends the current `_validate`):
unknown vocabulary terms; orphan regulations; double ownership; include cycles; guidance
without a catalogued binding target; second-phase layers in ACTIVE packs; scope dimensions not
allowed for the level; a pack referencing a class of another pack that it does not include;
department ids not in the core department list; no pack-id or brand literal in core code.

### 2.3 How the engine uses a pack (no rewrite)

The engine keeps its stage order. Three seams are introduced, each with the current AML
behaviour as the default implementation:

| Seam | Today | After |
|---|---|---|
| Actor / obliged-party lexicon | module constants in `applicability.py`, `entities.py` | `pack.actor_lexicon()`; AML pack returns the current tables unchanged |
| Risk categories, conflict lexicon, prompt examples | `engine.DEFAULT_CATEGORIES`, `conflict.ACTS/AUTHORITY`, `providers` few-shots | `pack.risk_categories()`, `pack.conflict_lexicon()`, prompt fragments versioned per pack in the prompt registry |
| Applicability target | `Company` (flat) | `EnterpriseProfile` targets; each legal entity still projects to `Company` (`tr/profile.to_company`) so the existing gate chain runs per entity |

Beverage applicability is computed by the core `ScopeResolver` (today `tr/routing.py`), then
narrowed at clause level by the extracted obligation's subject/conditions/exceptions matched
through the pack lexicon. The AML `rule_chain` remains the AML pack's clause gate; it is not
applied to beverage packs (its obliged-list logic is 5549-specific).

---

## 3. Migration plan

| Step | Change | Behaviour change | Proof |
|---|---|---|---|
| M-A | Generalise `tr/packs.Registry` → `PackRegistry` (per-pack taxonomy files, module composition, entry-point discovery). Keep `regchain.tr` imports as thin re-exports. | none | existing 88 pack tests + new pack-contract tests |
| M-B | Introduce the three seams with **AML defaults**; create `FINANCIAL_SERVICES_TR` pack whose hooks return the current constants by reference. Engine gains `pack=` parameter defaulting to the AML pack. | none | full AML suite unchanged; new equivalence test: `analyze()` on `backend/tests/fixtures/tedbirler-snapshot.json` produces byte-identical packets with and without the explicit pack |
| M-C | Move horizontal law out of `TR_FOOD_BEVERAGE_COMMON` into horizontal modules; expand taxonomies (§4.2). | beverage routing only | contrast tests updated deliberately, each change reviewed |
| M-D | `ingestion/repository.py`: regulator from source identity instead of `'FCA'` (SQL migration adding TR regulators); register mevzuat parser in `parse.parse`; Resmî Gazete document adapter. | FCA path identical | ingestion + postgres tests unchanged; new TR ingestion tests |
| M-E | Catalogue verification: each `RegulationMeta` moves `UNVERIFIED → VERIFIED` only after its text is fetched, its number/title match, and `source_hash` is recorded. | metadata only | verification tests; no hand-typed hashes |
| M-F | Extracted scopes: extraction output → `ObligationScope(origin=EXTRACTED, provision_ref=…)` + conditions/exceptions as predicates; explanation record; decision stages; guidance conflict. | new beverage outputs | dev-set evaluation (not gold) |
| M-G | Obligation mapping (department / document / control / evidence / status). | new | unit + dev-set |
| M-H | Change impact to business level. | new | unit + replay on two real versions of one regulation |
| M-I | Evaluation extensions + datasets + expert-gold workflow. | reports only | evaluation tests |

Constraint for every step: targeted tests → full backend suite (failure set must equal the
recorded baseline) → commit. No push without explicit approval.

---

## 4. Data schemas

### 4.1 Enterprise profile (extends `tr/profile.EnterpriseProfile`)

Structured graph (exists) plus two additions:

```jsonc
{
  "format": "cardaman-tr-enterprise-profile/2",
  "group": {"group_id": "…", "name": "…"},
  "legal_entities": [ { "entity_id": "…", "roles": ["PARENT","MANUFACTURING","DISTRIBUTION","SALES","MARKETING","IMPORT_EXPORT"],
                        "entity_classes": [...], "activity_classes": [...], "licenses": [...],
                        "product_ids": [...], "sales_channels": [...], "profile_complete": false } ],
  "facilities":  [ { "facility_id": "…", "facility_classes": ["BREWERY"|"BOTTLING_PLANT"|"MANUFACTURING_PLANT"|"WAREHOUSE"|
                                                               "DISTRIBUTION_CENTER"|"OFFICE"|"LABORATORY"|…],
                     "city": "…", "activities_complete": true } ],
  "products":    [ { "product_id": "…", "product_class": "…",
                     "attributes": {                       // typed, declared by the pack taxonomy
                       "abv_percent":        {"value": 0.4, "status": "STATED"},
                       "caffeine_mg_per_l":  {"status": "UNKNOWN"},
                       "added_sugar":        {"value": true, "status": "STATED"},
                       "imported":           {"value": false, "status": "STATED"}
                     },
                     "components": [],                     // for PROMOTIONAL_BUNDLE: product_ids it contains
                     "tags_complete": false } ],
  "activities":  [ ... ],
  "declared_facts": {                                      // optional, operator-entered, tri-state only
    "ecommerce": "UNKNOWN"
  },
  "products_complete": true
}
```

Flat facts (the form in the brief) are a **derived view**, not the source of truth:

```python
def profile_facts(profile, pack) -> dict[str, FactRecord]
# FactRecord = {value: YES|NO|UNKNOWN, derived_from: [json paths], declared: YES|NO|UNKNOWN|None}
```

- Each fact is defined in the pack taxonomy as a predicate over the graph, e.g.
  `alcoholic_beverage_manufacturer = ANY entity WITH activity PRODUCTION AND product alcohol_category ALCOHOLIC`.
- Evaluated with Kleene logic: an incomplete list can make a fact `UNKNOWN`, never `NO`.
- A declared fact that contradicts the derived value is a validation error (`PROFILE_FACT_CONFLICT`), not an override.
- `YES_OR_UNKNOWN` (as in the brief's Coca-Cola-type example) is not a value: it is stored as `UNKNOWN` until a product or activity record makes it `YES`. Values are never invented.

### 4.2 Taxonomy (per pack, hierarchical)

```jsonc
{
  "product_classes": {
    "ALCOHOLIC_BEVERAGE": {"parent": null, "label_tr": "Alkollü içki"},
    "BEER":               {"parent": "ALCOHOLIC_BEVERAGE"},
    "MALT_BEVERAGE":      {"parent": null, "alcohol": "BY_ABV"},
    "LOW_ALCOHOL_BEER":   {"parent": "BEER", "alcohol": "BY_ABV"},
    "ALCOHOL_FREE_BEER":  {"parent": null, "alcohol": "BY_ABV"},
    "PROMOTIONAL_BUNDLE": {"parent": null, "alcohol": "FROM_COMPONENTS"}
  },
  "alcohol_threshold": {"attribute": "abv_percent", "op": "gt", "value": null,
                        "source": {"regulation_id": "TR:KANUN:4733", "provision_ref": null}, "status": "UNVERIFIED"},
  "attributes": {"abv_percent": {"type": "number", "unit": "% vol"},
                 "caffeine_mg_per_l": {"type": "number", "unit": "mg/L"},
                 "added_sugar": {"type": "bool"}, "sweetener": {"type": "bool"},
                 "zero_sugar_claim": {"type": "bool"}, "imported": {"type": "bool"}},
  "facts": {"alcoholic_beverage_manufacturer": {"predicate": {...}}, "...": {}},
  "departments": ["LEGAL","COMPLIANCE","QUALITY","FOOD_SAFETY","MANUFACTURING","SUPPLY_CHAIN","PROCUREMENT",
                  "MARKETING","HR","DATA_PRIVACY","SUSTAINABILITY","ENVIRONMENT","SALES"]
}
```

- Alcohol category of a product is derived: `ALCOHOLIC`, `NON_ALCOHOLIC` or `UNKNOWN` (ABV missing for a `BY_ABV` class, or threshold `UNVERIFIED` and ABV near it). Low-alcohol and alcohol-free beer are therefore decided by a stated attribute and a cited threshold, not by a name.
- Bundles take the union of their components' categories; a bundle containing any alcoholic component is alcoholic for alcohol scopes.
- Alcohol pack additions: activities `BREWING`, `PACKAGING`, `HORECA_SUPPLY`, `DIGITAL_MARKETING`, `EVENTS`; facilities `OFFICE`, `LABORATORY`; product classes above; attribute `imported`.
- Non-alcohol pack additions: `FRUIT_DRINK`, `CONCENTRATE`, `SYRUP` (B2B), `MANUFACTURING_PLANT`, `LABORATORY`, `OFFICE`; "sweetened", "zero-sugar", "caffeinated" are **attributes**, not classes (a zero-sugar cola is a carbonated soft drink with `sweetener=true`, `zero_sugar_claim=true`).
- Regulatory domains become `obligation_categories` per pack (alcohol: production, sales, supply, presentation, distribution, licensing, advertising, promotion, sponsorship, age restriction, labelling, …; non-alcohol: hygiene, ingredients, additives, contaminants, microbiological criteria, nutrition information, sugar, sweeteners, caffeine, energy drinks, bottled water, traceability, recall, plastic packaging, recycling, …). Horizontal categories (KVKK, OHS, employment, competition, consumer, e-commerce, environment, waste, wastewater, emissions, water use) are owned by horizontal modules.

### 4.3 Obligation scope with condition and exception (extends `tr/core.ObligationScope`)

```jsonc
{
  "scope_id": "…", "regulation_id": "…", "version_id": "…",
  "level": "LEGAL_ENTITY|FACILITY|PRODUCT|ACTIVITY",
  "provision_ref": "md.5/f.1/b.a" , "provision_status": "RESOLVED",
  "origin": "EXTRACTED", "obligation_id": "…",            // link to the extracted, grounded obligation
  "actor":     {"classes": ["RETAILER"], "quote": "…", "span": [s, e]},
  "activity_classes": [...], "facility_classes": [...], "product_classes": [...],
  "alcohol_scope": "ALCOHOLIC",
  "jurisdiction": {"level": "NATIONAL"},                 // or {"level": "PROVINCE", "cities": [...]}
  "conditions": [ {"condition_id": "C1", "quote": "…", "predicate": {"fact": "activity.channels", "op": "contains", "value": "ONLINE"}} ],
  "exceptions": [ {"exception_id": "E1", "quote": "…", "predicate": {...}, "effect": "EXEMPTS|NARROWS",
                   "status": "GROUNDED|UNVERIFIED"} ],
  "category": "ALCOHOL_ADVERTISING", "scope_status": "DEFINED|UNCLEAR"
}
```

Predicate language (core, small and closed): `all`, `any`, `not`, `{fact, op, value}` with
ops `eq, in, contains, gt, ge, lt, le, exists`. Evaluated in Kleene logic over profile facts;
a predicate the core cannot evaluate from the profile yields `UNKNOWN` with
`CONDITION_UNEVALUABLE`. An exception with `UNVERIFIED` status can only move a decision to
`UNKNOWN` (`EXCEPTION_POSSIBLE`), never to `DOES_NOT_APPLY`.

### 4.4 Applicability decision (explained)

```jsonc
{
  "decision_id": "…", "stage": "ROUTED|GROUNDED|REVIEWED",
  "pack": "BEVERAGE_ALCOHOL_TR@0.2.0", "scope_id": "…",
  "source": {"regulation_id": "…", "version_id": "…", "source_class": "BINDING", "source_hash": "…",
             "provision_ref": "md.5/f.1", "quote": "…", "span": [s, e]},
  "target": {"level": "ACTIVITY", "id": "…", "entity_id": "…", "facility_id": null, "product_ids": ["…"]},
  "actor":     {"required": ["…"], "matched": "…", "fact": "legal_entities[2].entity_classes", "result": "YES"},
  "activity":  {"required": [...], "matched": "…", "fact": "…", "result": "YES"},
  "product":   {"required": {...}, "matched": ["…"], "not_matched": ["…"], "facts": ["products[0].attributes.abv_percent"], "result": "YES"},
  "facility":  null,
  "jurisdiction": {"result": "YES"},
  "conditions": [{"condition_id": "C1", "result": "UNKNOWN", "facts": ["…"]}],
  "exceptions": [{"exception_id": "E1", "result": "NO", "facts": ["…"]}],
  "status": "APPLIES|PARTIAL|DOES_NOT_APPLY|UNKNOWN",
  "reason_codes": ["ACTIVITY_MATCH", "ALCOHOL_SCOPE_MATCH"],
  "creates_obligation": true,
  "review_required": false, "review_reasons": []
}
```

Invariants (validator, core):
- `stage != ROUTED` ⇒ `source.provision_ref`, `quote`, `span` present and the quote is an exact substring of the stored version (existing grounding code).
- `status == DOES_NOT_APPLY` ⇒ at least one dimension `result == NO` resting on a **complete** fact or a GROUNDED exception.
- `creates_obligation` ⇒ `source_class == BINDING`.
- Guidance vs binding disagreement ⇒ binding decision kept, `review_required`, `GUIDANCE_CONFLICT`.
- Every `fact` path resolves in the profile version recorded with the decision.

### 4.5 Obligation mapping (policy / control / evidence / gap)

```jsonc
{
  "internal_documents": [{"document_id": "SOP-LABEL-01", "type": "POLICY|SOP|STANDARD|WORK_INSTRUCTION|SPECIFICATION",
                          "owner_department": "QUALITY", "applies_to": {"entity_ids": [...], "facility_ids": [...], "product_ids": [...]},
                          "version": "…", "hash": "…"}],
  "controls": [{"control_id": "CTL-LABEL-APPROVAL", "description": "Legal and Quality approval before label release",
                "owner_department": "QUALITY", "type": "PREVENTIVE|DETECTIVE", "frequency": "PER_RELEASE",
                "documents": ["SOP-LABEL-01"], "evidence_types": ["APPROVED_LABEL_ARTWORK"]}],
  "evidence": [{"evidence_id": "…", "control_id": "…", "type": "APPROVED_LABEL_ARTWORK", "period": "…", "hash": "…"}],
  "mappings": [{"decision_id": "…", "obligation_id": "…", "departments": ["QUALITY","LEGAL","MARKETING"],
                "department_source": "PACK_DEFAULT|COMPANY_OVERRIDE",
                "document_coverage": "COVERS_TEXT|PARTIAL|CONFLICT|NO_EVIDENCE|UNKNOWN",   // existing engine output
                "control_ids": [...], "evidence_ids": [...],
                "status": "COVERED|PARTIALLY_COVERED|NOT_COVERED|CONTRADICTED|UNKNOWN", "reasons": [...]}]
}
```

Status is counted, not asked of a model:

| document_coverage | control mapped | evidence present | status |
|---|---|---|---|
| CONFLICT | any | any | CONTRADICTED |
| COVERS_TEXT | yes | yes, in period | COVERED |
| COVERS_TEXT | yes | no / stale | PARTIALLY_COVERED (`EVIDENCE_MISSING`) |
| COVERS_TEXT | no | — | PARTIALLY_COVERED (`CONTROL_MISSING`) |
| PARTIAL | any | any | PARTIALLY_COVERED |
| NO_EVIDENCE | no | — | NOT_COVERED |
| UNKNOWN or decision UNKNOWN | any | any | UNKNOWN |

Only decisions with `status in (APPLIES, PARTIAL)` are mapped; `PARTIAL` mappings are scoped
to the decision's matched products/facilities.

### 4.6 Change impact

```jsonc
{
  "change_id": "…", "regulation_id": "…", "from_version": "…", "to_version": "…",
  "provision_changes": [{"provision_ref": "md.7/f.2", "kind": "ADDED|REMOVED|MODIFIED",
                         "impact_types": ["LABEL_CONTENT"], "quantities": [{"old": "…", "new": "…"}]}],
  "affected": [{"obligation_id": "…", "decision_before": "APPLIES", "decision_after": "APPLIES",
                "delta": "NEWLY_APPLIES|NO_LONGER_APPLIES|CHANGED_REQUIREMENT|UNCHANGED|UNKNOWN",
                "entities": [...], "facilities": [...], "products": [...],
                "documents": [...], "controls": [...], "departments": [...],
                "actions": [{"action": "UPDATE_DOCUMENT", "target": "SPEC-ENERGY-LABEL", "department": "QUALITY"}]}]
}
```

Chain: version chain (`tr/core.version_chain`) → provision diff (`ingestion/diff`, `compare`)
→ re-extract changed provisions only (existing carry-forward in `impact.py`) → re-resolve scopes
old vs new per target → join with mappings → actions. Products are reached through
`product_ids` on decisions; facilities through `facility.product_ids`; documents through
`applies_to`; departments through mappings. A change with no reachable target is reported as
`NO_TARGET_AFFECTED`, never silently dropped.

---

## 5. Source adapter plan

```python
class SourceAdapter(Protocol):
    adapter_id: str                                  # 'mevzuat', 'resmi-gazete', 'tarimorman-gkgm', …
    hosts: frozenset[str]                            # added to fetch.ALLOWED_HOSTS
    def discover(self, query) -> list[SourceRef]: ...
    def fetch(self, ref) -> Download: ...            # existing fetch.py
    def parse(self, download) -> Document: ...       # existing parsers
    def identity(self, document) -> RegulationIdentity: ...  # regulator, type, number, publication date
```

| Adapter | Status | Work |
|---|---|---|
| mevzuat.gov.tr | parser exists (`ingestion/mevzuat.py`) | register in `parse.parse` and the CLI; store versions under TR regulators |
| Resmî Gazete | title scan only (`pilot/turkiye.py`) | document fetch + parse; publication date as version anchor; link to mevzuat consolidated text |
| Ministry / regulator guidance (Tarım ve Orman, Ticaret / Reklam Kurulu, Sağlık, Çevre, KVKK, Rekabet, Hazine ve Maliye) | none | one adapter per host; ingested as `OFFICIAL_GUIDANCE`, must name the binding text it interprets |
| Board decisions (Reklam Kurulu, Rekabet Kurulu, KVKK Kurulu) | none | `DECISION_PRECEDENT`, second phase, never enabled in an ACTIVE MVP pack |
| FCA handbook / PDF | exists | unchanged |

Pack catalogues reference regulations by `regulation_id`; the adapter supplies bytes, versions
and hashes. A pack cannot mark a catalogue entry VERIFIED by itself.

---

## 6. Evaluation plan

### 6.1 Datasets

| Dataset | Labels | May be reported as |
|---|---|---|
| `BEVERAGE_TR_DEV_V1` | developer-labelled on synthetic pilot profiles, each label with a written reason and source quote | development signal only, never accuracy |
| `BEVERAGE_TR_EXPERT_GOLD_V1` | two blind expert reviews + adjudication (existing `expert_gold` workflow), frozen | accuracy, once the sample target is met |

Targets: ≥ 100 independent reviewed examples per critical task per pack; ≈ 300 positive
examples for contradiction. Below target the task reports `INSUFFICIENT_SAMPLE` with `n`,
target and the metric values shown as indicative only. No label is inferred, copied from a
model, or recreated from lost artefacts.

### 6.2 Tasks and metrics

Tasks: `APPLICABILITY` (APPLIES / PARTIAL / DOES_NOT_APPLY / UNKNOWN), `ACTOR_MATCH`,
`PRODUCT_MATCH`, `FACILITY_MATCH`, `ACTIVITY_MATCH`, `EXCEPTION`, `OBLIGATION_EXTRACTION`,
`CONTRADICTION`, `POLICY_COVERAGE`, `SOURCE_GROUNDING`.

For each task and each pack: precision, recall, F1 (per class, one-vs-rest, plus macro),
false-positive rate, false-negative rate, and **unsupported conclusion count** = decisions
with a non-UNKNOWN status that (a) cite no resolved provision, (b) cite a quote that is not an
exact substring of the stored version, or (c) rest on a fact path that does not resolve or is
UNKNOWN. UNKNOWN is scored as its own class and additionally as abstention rate; a `NO` where
gold is `UNKNOWN` counts as an unsupported conclusion.

Code changes needed in `evaluation/metrics.py`: `false_negative_rate` field, per-class
one-vs-rest rates, unsupported count, `INSUFFICIENT_SAMPLE` status per task; in
`expert_gold.py`: the matching tasks above; in `taxonomy.py`: dispute tables keyed by dataset
id instead of `tr-aml-v1` only.

### 6.3 Contrast families (paired cases)

| Family | Pair | Expected |
|---|---|---|
| Cross-pack | same regulation, alcohol group vs non-alcohol group | differ where the scope is alcohol-specific (advertising, sales/presentation); agree on horizontal duties (packaging waste, labelling, KVKK) |
| Cross-product | same entity, beer vs alcohol-free beer vs low-alcohol beer (ABV stated / missing) | alcohol scopes APPLIES / DOES_NOT_APPLY / UNKNOWN |
| Cross-product | same bottler, energy drink vs water vs zero-sugar cola | energy/caffeine, water and claims scopes separate |
| Cross-entity | production vs sales vs marketing entity of one group | sales and advertising duties land on different entities |
| Cross-facility | brewery / bottling plant vs warehouse / office / laboratory | facility duties land on production sites only |
| Completeness flip | same case with a list marked incomplete | DOES_NOT_APPLY must become UNKNOWN, never stay NO |
| Exception | same case with the exception predicate true / false / unknown | EXEMPTS / APPLIES / UNKNOWN |
| Authority | guidance-only duty; guidance contradicting binding | no obligation; binding wins + review |

Reported metric: **pair consistency** = share of pairs whose two outcomes both match gold.

---

## 7. Test plan

| Layer | Tests | Rule |
|---|---|---|
| AML contract | all existing non-`tr` suites | unchanged; failure set must equal the recorded baseline until artefacts are restored |
| AML equivalence (M-B) | `analyze()` with explicit `FINANCIAL_SERVICES_TR` pack vs default on `tests/fixtures/tedbirler-snapshot.json` | byte-identical packet |
| Architecture | core modules import no `packs.*`; no pack id or brand literal in core; hooks are pure (no I/O) | static test |
| Pack contract (parametrised over every discovered pack) | manifest, taxonomy, lexicon, catalogue, scopes, exceptions, ownership, departments validate; single ownership; include closure; authority layers | runs automatically for future packs |
| Profile | graph integrity; derived facts; Kleene evaluation; declared-vs-derived conflict; ABV-derived alcohol category; bundles | unit |
| Resolver | per-dimension YES/NO/UNKNOWN; conditions; exceptions; jurisdiction; rollups; decision invariants (§4.4) | unit + property tests (random completeness flips never create NO) |
| Grounding | GROUNDED decisions' quotes are exact substrings; provision refs resolve in the stored version | unit |
| Mapping | status table (§4.5) exhaustively | unit |
| Change impact | two synthetic versions of one provision → expected affected products/facilities/documents/departments | unit; replay on a real version pair after M-D |
| Evaluation | new metrics, INSUFFICIENT_SAMPLE, unsupported count, pair consistency | unit |
| Contrast pilots | existing `test_tr_beverage_pilot.py`, extended with low-alcohol beer, bundle, laboratory/office, horizontal modules | unit (fixtures are synthetic; not evaluation) |

---

## 8. Implementation order

1. **Evaluation primitives** (cheap, gates everything after): FNR, one-vs-rest rates, unsupported count, `INSUFFICIENT_SAMPLE`, pair consistency. No dataset needed.
2. **M-A** PackRegistry generalisation + pack-contract and architecture tests.
3. **M-B** AML seams with defaults + `FINANCIAL_SERVICES_TR` pack + equivalence test. Stop and report if equivalence is not exact.
4. **Profile v2**: typed attributes, derived facts, Kleene predicates, ABV-derived category, bundles, entity roles, facility classes OFFICE / LABORATORY / MANUFACTURING_PLANT.
5. **M-C** horizontal modules; taxonomy expansion for both packs; pilots extended (standard / low-alcohol / alcohol-free beer; carbonated / zero-sugar / energy / water).
6. **Decision record v2** with stages, conditions, exceptions, jurisdiction, fact paths and invariants; `GUIDANCE_CONFLICT` routing to review.
7. **M-D / M-E** TR ingestion into the repository and catalogue verification (requires network access to mevzuat.gov.tr and the Resmî Gazete; nothing is marked VERIFIED without fetched text).
8. **M-F** extracted scopes: obligations from verified texts → scopes with provision refs, conditions and exceptions; beverage actor lexicon; `BEVERAGE_TR_DEV_V1` built from these, with reasons and quotes.
9. **M-G** obligation mapping (departments, documents, controls, evidence, status).
10. **M-H** change impact to business level.
11. **Expert gold** collection with reviewers you designate; metrics reported per pack with `INSUFFICIENT_SAMPLE` until targets are met.

Out of scope until asked: multi-tenant API, UI, deployment, fine-tuning, model changes, cloud LLMs.

## 9. Status, 1 October 2026: rules first, a model only where the rules cannot decide

Implemented on `feat/beverage-pilot-ai` (local commits, not pushed). Every number below is a developer's label scored
as INDICATIVE; the detail is in `evaluation/reports/beverage-tr-20261001/`.

### 9.1 What decides

| Step | Module | Who decides |
|---|---|---|
| clause reading (kind, addressee, exceptions, conditions, quantities) | `tr/frames.py`, `tr/extraction.py` | rules; the extraction model (qwen3:4b) reads the same units as a second reader and never decides |
| applicability | `tr/routing.py` | rules over the profile's facts (Kleene); a duty whose clause names no constraint applies through its regulation's presence in the pilot's catalogue and is recorded as `applicability_basis = REGULATION_SCOPE` |
| policy coverage | `tr/compare.py` (`tr-compare-rules-v2`) | rules, element by element: act, polarity, products, places, other party, limits, list items, label particulars, deadlines, time windows, conditions, exceptions |
| candidate statements | `tr/semantic.py` (`tr-candidates-v2`) | character n-grams, recorded bge-m3 similarities, an anchored lower floor; a candidate is never a coverage |
| adjudication | `tr/adjudicate.py` | qwen3:8b with thinking, only on escalated rows (a candidate the rules could not relate, a statement that may go against the duty, an open coverage, a contested reading); one answer per statement with an exact quote; a failed or truncated call is recorded, retried once without thinking, and leaves the row open |
| decision | `compare.combine_coverage`, `adjudicate.assess_obligation` | the rules' word stands. The model may fill PARTIAL where the rules found nothing; its COVERED and its CONTRADICTED are proposals with `REVIEW_REQUIRED`; an element check of the rules is never overruled; what a decision rests on is verified again (`adjudicate.verify`: exact spans of the stored version and of the documents in force, gates of the applicability) |

Three things are counted apart everywhere: automatic decisions (right or wrong), cases sent to a person (never counted as right: `strict_accuracy`), and the model's proposals on their own.

### 9.2 Measured

Rule reader on clauses: DEV 289/289 kind, 204/204 addressee, 60/60 exception (fit); HOLDOUT 105/112, 55/61, 22/27 (blind labels; the exception count was 21 before the exemption wording "gerek yoktur" was added on a corpus finding; five of the remaining misses are one disputed label family, Yönetmelik 14646 md. 22/3). Extraction model qwen3:4b: DEV 231/289, HOLDOUT 89/112.

Policy coverage, rules only: DEV 33 → 36 of 45; HOLDOUT 10 → 19 of 28 (18 misses read one by one: 11 statements retrieved but not related by wording, 7 not retrieved, no other error kind); VALIDATION (unseen, 42 cases on a second register per pilot, scored once) 17 → 19. Whole corpus CONTRADICTED rows unchanged by v2 (brewer 23, bottler 1, importer 7).

Rules + qwen3:8b thinking (run 8, commit 3e36ebf): see the report's table for the per-split rows (automatic right / wrong, review, proposals, seconds per call). On the unseen set: 25 of 42 right automatically, 6 wrong (one contradiction the rules invent — a limit written as a prohibition, "4,8'i geçemez", read as negating the duty — and one they miss, a fruit-ratio minimum the quantity reader has no attribute for), 11 to a person (10 of them cases the rules had wrong), 18 of 21 proposals right, no false COVERED, ~18 s per call. The no-thinking variant is five times faster and claims a false CONTRADICTED on about half of the DEV escalations; it is the fallback only.

Corpus: escalation 4.8 / 5.7 / 3.4 % of the obligations (6.2 / 10.0 / 7.7 % of the rows) for the three pilots at the frozen commit; the report holds the run-8 figures with model time per obligation and per call.

### 9.3 Known limits

- A limit written as a prohibition of the other polarity ("... geçemez" against "... olmalıdır") is read as a negation: one false CONTRADICTED on the unseen set (also present before v2). A property limit without a lexicon attribute (fruit ratio, quinine in mg/L) is not compared: one missed contradiction.
- Paraphrase without an element the rules can check goes to the model and then to a person; that is by design, and it is where the review load is.
- The clause escalation (nested exception, several parties, cross-reference, unclear addressee, reader disagreement) does not target the rule reader's errors: on HOLDOUT it reached 1 of 7.
- HOLDOUT is development data since commit 3e36ebf; the next round needs a fresh unseen set, and the validation register and its labels were written by the same developer (blind to results, not to the code).
- Applicability of horizontal duties rests on the catalogue, not on a company fact (`REGULATION_SCOPE`, 695 of 3235 brewer rows); it is recorded, not verified against the profile.
- The anchored lower retrieval floor (0.55 with two shared stems) admits far more pairs on the whole corpus than on the labelled cases: escalation rose to 19 / 31 / 25 % of the rows (13.5 / 17.1 / 12.2 % of the obligations), above the 5-15 % target for rows. Three shared stems measured 7/9 gold instead of 8/9 on HOLDOUT with 3 of 550 other pairs; not changed in this round so that the validation set stays scored once. Model time stayed low: 6.9 s per call, 0.95 s per obligation (brewer, 8B without thinking, 370 calls, 40 minutes).
- A PARTIAL the model reads alone is taken as the coverage (146 brewer rows); measured right on the labelled sets, not at corpus scale.
- Follow-up the same night (tr-compare-rules-v3, engines.py): limits written as prohibitions are limits (the invented contradiction is gone; rules only on the unseen set 19 -> 22 of 42), a property named in the text is the attribute of its limit, an unresolved attribute decides nothing; a PARTIAL only the model reads is a proposal; the anchored floor needs three shared stems (corpus escalation 9.7 / 17.0 % of the rows instead of 19 / 31 %, at the cost of one gold statement in nine on HOLDOUT); one SectorEngine per pack over shared ExpertServices (`python -m regchain.tr assess`). The validation set was read to make these changes and is development data from here on.
- Development labels only; no expert gold.

### 9.4 Follow-up, 2 October 2026: real public documents, comparer v4, sales channel and packaging dimensions

- Public codes of ethics of two brewers and a bottler's human-rights document, read as registers by `scripts/tr_external_test.py` (test input only, never committed), produced 26 CONTRADICTED rows on the brewer pilot, all invented from boilerplate words. `tr-compare-rules-v4`: a permission contradicts an outright ban only when it is about the duty's subject (its products, the company's brand, logo or trade name, or two of its distinctive words); a negation needs most of the duty's words and at least three; a one-word duty is related only by its elements; regulatory boilerplate is not distinctive wording. Labelled sets and corpus CONTRADICTED counts unchanged.
- `ObligationScope.sales_channels` is a dimension of entity, facility and activity scopes (`routing.GATE SALES_CHANNEL`): a duty that governs a sale, or is about distance selling, and names a channel (bakkal, büfe; süpermarket, zincir market; mesafeli sözleşme, e-ticaret, internet üzerinden satış) binds the entities that sell through that channel; one that sells through none is not bound; one whose channels are not stated answers UNKNOWN. A channel word inside a licence name, a web address or an exception is no channel. Packaging words (depozito, cam şişe, pet/plastik şişe, metal/alüminyum/teneke kutu) become `product_attributes` of product-level scopes only. Corpus: 83 scopes carry a channel (nearly all ONLINE on the e-commerce law and regulation), 4 carry a packaging.

### 9.5 Board decisions as a company-specific layer (`tr/decisions.py`, 2 October 2026)

A board decision binds the undertaking it names, not a class. The DECISION_PRECEDENT layer stays out of every active pack; `decisions.py` reads a stored decision (text + record with hash, institution allowlisted in `adapters.INSTITUTIONS`, e.g. REKABET on rekabet.gov.tr) beside the sector engines for the entities the profile names as its addressees (`LegalEntity.bound_by_decisions`, or `--addressee` on the CLI) and for no other entity. Its numbered commitment items are sections for the same clause splitter and rule reader as a regulation (commitment wording: future tense and "taahhüt eder" are duties; the authority's review is a task of the administration); every duty is bound to the addressee through a DECISION_ADDRESSEE gate; the comparer, candidates, adjudicator and verification run unchanged over a `LayeredStore` (corpus + decisions); every row is REVIEW_REQUIRED (`DECISION_PRECEDENT_REVIEW`) because the decision reader has no labelled cases. Import: `scripts/tr_decision_import.py`; run: `python -m regchain.tr decisions`. The addressee is a profile fact; nothing in the code names a company.

### 9.6 QDMS export with a human approval gate (`tr/qdms.py`, 2 October 2026)

Regulation -> Cardaman reasoning -> impacted policy/control -> human approval -> QDMS action. Every gap row (sector engines and the decision layer) becomes a QDMS row grouped as entity, provision, applicability, reason, impacted process, policy status, required actions, evidence and human approval. Gap actions map onto a generic vocabulary (DOCUMENT_CHANGE_REQUEST, CONTROL_DEFINITION_REQUEST, EVIDENCE_REQUEST, COMPLIANCE_REVIEW_TASK, NONCONFORMITY); a row with an open decision leads with a review task. Every action is DRAFT and every row PENDING on export; an approvals file binds each decision to the row's fingerprint (reasoning, actions and the version of the text), so an approval of a row that changed since is refused as stale. Only approved rows' actions become READY_FOR_QDMS. Nothing is sent: the output is JSON and CSV files; mapping onto a given QDMS product's import format, and grouping the per-duty change requests into per-document requests, are open.

---

## Open decisions (owner: Emir)

1. Whether the AML knowledge becomes `FINANCIAL_SERVICES_TR` now (step 3) or stays as the engine default until a second financial jurisdiction appears.
2. Who the expert reviewers for `BEVERAGE_TR_EXPERT_GOLD_V1` are (two blind reviewers + adjudicator per task).
3. Whether network access to mevzuat.gov.tr / resmigazete.gov.tr / ministry sites may be used for step 7.
