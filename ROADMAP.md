# Product Roadmap: TTRPG Studio

## North Star
Build a **t3.chat-like operating workspace** for TTRPG teams that can run from solo creator mode to enterprise-grade multi-team production.

## Guiding principles
- **Chat-first orchestration**: every output can become structured data that feeds Foundry, World Bible, and Prompt Forge.
- **Human-in-the-loop reliability**: generated data is reviewable, editable, and traceable.
- **Enterprise trust posture**: security, observability, governance, and auditability are first-class.

## Phase 1 — Workflow Integration Foundation (0-6 weeks)
- Add structured extraction pipeline from assistant replies into reusable context objects.
- Add "apply to" bridges so chat output can prefill:
  - Project Foundry fields
  - World Bible drafts
  - Prompt Forge prompt fields
- Introduce shared context-pack assembly for generation calls.
- Add field-level confidence indicators and user correction before save.
- Expand regression checks around prompt building and import/export behavior.

**Exit criteria**
- Users can go from one chat answer to a saved project blueprint + world entry + prompt vault item in under 60 seconds.

## Phase 2 — Domain Model + Memory (6-12 weeks)
- Formalize canonical entities (factions, NPCs, locations, events, timelines, items).
- Add entity extraction with source citation links back to chat/world entries.
- Add retrieval layer (hybrid keyword + vector + graph neighborhood).
- Add conflict detection (lore contradictions, duplicate entities, timeline drift).
- Introduce campaign workspaces with persistent IDs (beyond session state).

**Exit criteria**
- Model outputs can cite source memory and pass basic continuity checks before publishing.

## Phase 3 — Platform & Data Architecture (12-20 weeks)
- Refactor single-file app into modular architecture:
  - `ui/`, `services/`, `domain/`, `storage/`, `observability/`
- Add persistent storage migration path:
  - local SQLite (dev) -> managed Postgres (prod)
- Add async job layer for long-running generations and bulk exports.
- Add versioned data contracts (schema migrations + backward compatibility checks).
- Introduce cache strategy for retrieval, token spend computations, and prompt templates.

**Exit criteria**
- Zero data loss during migration rehearsal; deterministic rollback path documented.

## Phase 4 — Enterprise Security & Governance (20-30 weeks)
- Add authentication and SSO (OIDC/SAML options).
- Add RBAC/ABAC roles (admin, editor, reviewer, read-only).
- Encrypt secrets and sensitive campaign data at rest and in transit.
- Add tenant/workspace isolation and environment profiles.
- Add immutable audit events for content lifecycle actions.
- Add policy controls for model/provider usage and prompt redaction.

**Exit criteria**
- Security review and threat model completed with remediation tracking.

## Phase 5 — Enterprise Operations & Reliability (30-40 weeks)
- Add centralized telemetry: latency, error taxonomies, token spend, provider health.
- Add SLOs/SLAs and alerting for degraded model routes.
- Add backup/restore, disaster recovery, and retention policies.
- Add feature flags and staged rollouts (dev/stage/prod).
- Add incident playbooks and operational dashboards.

**Exit criteria**
- Meets target SLOs for generation success and workspace load times.

## Phase 6 — Advanced Team Intelligence (40+ weeks)
- Multi-agent orchestration for campaign pipelines (lore QA, encounter QA, art direction QA).
- Prompt Forge evolution into reusable style systems and branded content packs.
- Workflow automation (publish-ready packets, weekly recaps, GM prep digests).
- Marketplace-style template packs and org-level shared libraries.
- API + webhook ecosystem for external VTT and content pipeline integrations.

**Exit criteria**
- Teams can automate recurring prep workflows and integrate into external systems.

## Cross-cutting quality bars (every phase)
- Strong typing and schema validation around generated artifacts.
- Test coverage for critical workflows and data migrations.
- Backward-compatible exports/imports and reproducible environments.
- Accessibility and UX consistency across tabs.
- Cost governance guardrails with budget alerts.
- Documentation for developers, operators, and end users.
