# Product Roadmap: TTRPG Studio

## North Star
Build a **t3.chat-like** interface specialized for TTRPG creators: worldbuilding, campaign planning, image prompt workflows, and dependable everyday usage.

## Phase 1 — Foundation Hardening (near term)

- Stabilize current multi-provider chat UX (OpenAI + OpenRouter)
- Extract app into modules (`ui/`, `services/`, `storage/`, `domain/`)
- Add typed config layer for providers/models
- Add regression tests for prompt-building and import/export behavior
- Add robust error taxonomy (auth errors, quota, network, model not found)

## Phase 2 — Data + Admin Operations

- Add authentication (single-user local first, then team auth)
- Add role-based access control (admin/editor/viewer)
- Migrate from session state to database persistence (SQLite -> Postgres)
- Build admin dashboard: usage analytics, logs, workspace backups
- Add project-level settings and environment profiles

## Phase 3 — TTRPG Domain Intelligence

- Canonical world model:
  - factions
  - NPCs
  - locations
  - timelines
  - events
- Consistency checks (detect lore conflicts)
- Session prep assistants (encounters, hooks, pacing plans)
- Campaign memory retrieval with citations to source notes

## Phase 4 — Image Workflow Expansion

- Keep current prompt vault and add metadata tagging/search
- Integrate image generation providers (OpenAI, OpenRouter-routed models, others)
- Add batch generation + versioned prompt experimentation
- Store image assets with links to campaign entities

## Phase 5 — Daily-use Productivity Layer

- Recurring reminders, checklist templates, and session agendas
- "Today" dashboard (active campaign, next tasks, latest lore changes)
- Offline-first drafting mode + sync
- Notification hooks (email/Discord/webhooks)

## Quality bars for every phase

- Clear, commented code
- Tests for core workflows
- Migration-safe storage changes
- Observability: spend, latency, reliability
- Security reviews for any auth or secrets features
