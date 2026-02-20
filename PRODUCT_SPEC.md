# TTRPG Studio — Product Specification (v2)

## Purpose
This document is the single source of truth for the next implementation cycle. It consolidates the work-session decisions around:
- Chat-to-workspace orchestration
- World Bible becoming a reliable canonical system
- Prompt Forge evolving into a generation engine
- Enterprise-readiness requirements

---

## 1) Product Vision
TTRPG Studio is a chat-first operating workspace where a single assistant response can be transformed into structured, reviewable campaign assets across:
- Project Foundry (campaign plans)
- World Bible (canonical memory)
- Prompt Forge (creative production prompts)

### Success statement
A GM or content team can move from idea -> reviewed canon -> playable outputs -> production-grade prompts with full traceability and minimal re-entry.

---

## 2) Scope for the next build cycle

## In scope
1. Structured context extraction from chat into typed artifacts.
2. World Bible overhaul for clarity, structure, and reliability.
3. Prompt Forge overhaul into a reusable prompt engine.
4. Improved cross-tab context pack assembly.
5. Spec-aligned roadmap and acceptance criteria.

## Out of scope (this cycle)
1. Full multi-tenant production auth stack.
2. Complete database migration (SQLite/Postgres implementation can begin but not be finished in this cycle).
3. External marketplace ecosystem.

---

## 3) Shared architecture concepts

## 3.1 Structured Context Artifact (SCA)
Every extraction from chat creates an artifact with immutable metadata.

Required fields:
- `id` (UUID)
- `created_at`
- `source_message_ref`
- `workspace_ref`
- `extraction_version`
- `confidence_summary`
- `payload`

### Payload domains
- Foundry seeds (`project_name`, premise, themes, factions, locations)
- World seeds (candidate entities, facts, timeline hints, tags)
- Prompt seeds (subject, style, mood, constraints, negatives)

## 3.2 Context Pack
A normalized request context assembled before generation actions.

Sources:
1. Active chat intent
2. Approved World Bible canon
3. Selected Foundry blueprint/project
4. Prompt Forge recipe profile

All generation endpoints should consume a context pack (not ad-hoc concatenation only).

---

## 4) World Bible Specification (functional + clear)

## 4.1 Data model
World Bible must support two layers:
1. **Entries** (human-authored notes, drafts, logs)
2. **Canonical Entities** (typed records)

### Entity types (required)
- Faction
- NPC
- Location
- Event
- Timeline Arc
- Item/Artifact

### Common entity fields
- `entity_id`
- `type`
- `name`
- `aliases[]`
- `status` (`draft`, `reviewed`, `canon`, `deprecated`)
- `summary`
- `facts[]` (structured claims)
- `tags[]`
- `source_refs[]` (entry/chat/provenance links)
- `updated_at`, `updated_by`

## 4.2 Editorial lifecycle
Required workflow:
1. Draft capture
2. Review queue
3. Canon approval
4. Change log entry

Each transition must emit an audit event.

## 4.3 Conflict detection
System must detect and surface potential conflicts:
- Name collisions (same/near-duplicate entities)
- Fact contradiction (e.g., timeline inconsistency)
- Status conflicts (deprecated entity referenced as canon)

Resolution actions:
- Merge
- Fork
- Supersede
- Ignore (with reason)

## 4.4 Retrieval behavior
When World Bible context is injected into prompts:
- Retrieve by hybrid strategy (tag + lexical + semantic + graph neighbor)
- Return citations to source entries/entities
- Enforce context budget with deterministic trimming

## 4.5 UI requirements
World Bible area should be split into clear sub-workflows:
1. Capture (new draft)
2. Browse (filters/table)
3. Graph (entity relationships)
4. Conflicts (triage queue)
5. Timeline (ordered events)

---

## 5) Prompt Forge Specification (engine upgrade)

## 5.1 Core concept
Prompt Forge becomes a composable system driven by versioned recipes, not a single joined prompt string.

## 5.2 Prompt Recipe model
Each recipe includes:
- `recipe_id`, `version`
- `use_case` (concept art, map, portrait, scene, handout)
- `template_blocks[]`
- `required_slots[]`
- `optional_slots[]`
- `style_profile`
- `negative_policy`
- `provider_adapters`

## 5.3 Multi-pass pipeline
Required passes:
1. **Concept pass** (intent and subject)
2. **Enrichment pass** (style, composition, world details)
3. **Constraint pass** (negatives, safety, output formatting)
4. **Provider adaptation pass** (model/provider syntax if needed)

## 5.4 Variant generation
Engine should support generating N variants from one recipe run.

Each variant stores:
- rendered prompt text
- key slot values
- recipe version
- context pack hash
- optional score metrics

## 5.5 Quality scoring (initial)
Score dimensions:
- Lore consistency
- Visual specificity
- Composition clarity
- Reusability

## 5.6 Asset linkage
Saved prompt variants can be linked to:
- World Bible entities
- Foundry project records
- Session outputs

---

## 6) Chat, Foundry, World Bible, and Prompt Forge connectivity

## 6.1 Minimum required flow
1. User chats with assistant.
2. Assistant output can be extracted into SCA.
3. User reviews SCA and applies selected fields.
4. Applied data pre-fills Foundry, World Bible drafts, Prompt Forge recipes.
5. User saves/approves outputs; provenance links are retained.

## 6.2 Traceability requirement
Every saved artifact should answer:
- Where did this come from?
- Which source message/entity informed it?
- Which version of extraction/recipe built it?

---

## 7) Non-functional enterprise requirements

## 7.1 Security and governance
- Auth + SSO capability path (OIDC/SAML)
- Role model path (admin/editor/reviewer/viewer)
- Encryption in transit and at rest
- Immutable audit logs for critical events

## 7.2 Reliability and observability
- Structured error taxonomy
- Metrics for latency, token usage, success/failure rates
- Provider health visibility
- Backup/restore and DR planning

## 7.3 Data and migration readiness
- Versioned schemas for artifacts/entities/recipes
- Migration-safe export/import behavior
- Backward compatibility checks during load

---

## 8) Acceptance criteria (session close)
This specification is considered implemented when:
1. World Bible supports typed entities with lifecycle states and conflict triage.
2. Prompt Forge supports recipe-driven multi-pass generation and variants.
3. Chat extraction artifacts can be reviewed and selectively applied across tabs.
4. Generated outputs include source citations/provenance links.
5. Export/import preserves new artifact, entity, and recipe structures.

---

## 9) Implementation sequence (recommended)
1. Data contracts: entity + recipe + SCA schemas.
2. World Bible UI/data model split (entries vs entities).
3. Conflict detection and review queue.
4. Prompt recipe runtime + variant storage.
5. Context pack service and citation injection.
6. Observability and policy guardrails.

---

## 10) Handoff notes
- Keep this spec updated as authoritative design intent.
- Any roadmap or code change should reference relevant section numbers from this document.
- If behavior diverges from spec, either update implementation or submit explicit spec amendment.
