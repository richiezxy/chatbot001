"""Immersive TTRPG Studio — Streamlit single-file app.

Includes:
  1) World Bible -> searchable mini "knowledge graph"
  2) Token tracking + model spend meter (best-effort; exact when usage returned)
  3) Project Foundry -> auto-generate Session 1 content from a blueprint

Future multi-file refactor notes (NOT required now):
- /core: provider client, token accounting, prompt builders
- /kg: entity extraction, graph building, graph UI
- /storage: SQLite/Postgres persistence + user workspaces
- /prompts: templates (session1, factions, etc.)
- /ui: theme / CSS

Requirements:
  pip install streamlit openai
Optional:
  pip install tiktoken  (improves token estimates)
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional, Tuple

import openai
import streamlit as st

# ============================================================
# App bootstrap (MUST be first st.* call)
# ============================================================
st.set_page_config(page_title="TTRPG Studio", page_icon="🧙", layout="wide")


# ============================================================
# Constants / Defaults
# ============================================================
OPENROUTER_MODEL_OPTIONS = [
    "openai/gpt-4o-mini",
    "stepfun/step-3.5-flash:free",
    "arcee-ai/trinity-large-preview:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "arcee-ai/trinity-mini:free",
    "qwen/qwen3-vl-30b-a3b-thinking",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "z-ai/glm-4.5-air:free",
    "google/gemma-3n-e2b-it:free",
    "deepseek/deepseek-r1-0528:free",
    "qwen/qwen3-4b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

DEFAULT_COSTS_PER_1M = {
    # Placeholders; pricing changes by provider/model/account.
    # Set your own numbers in the sidebar for accurate estimates.
    "prompt_usd_per_1m": 0.50,
    "completion_usd_per_1m": 1.50,
}


# ============================================================
# Session State
# ============================================================
def ensure_state() -> None:
    defaults: Dict[str, Any] = {
        "messages": [],
        "world_bible": [],
        "prompt_vault": [],
        "audit_log": [],
        "project_blueprints": [],
        # Knowledge graph caches
        "kg_nodes": {},  # name -> {"type": str, "count": int}
        "kg_edges": {},  # (a,b) -> weight
        "kg_built_at": None,  # iso timestamp
        # Token & cost tracking
        "usage_total": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "by_model": {},  # model -> dict
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ensure_state()


# ============================================================
# Utilities
# ============================================================
def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(event: str) -> None:
    st.session_state.audit_log.append(f"[{utc_ts()}] {event}")


def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


# ============================================================
# Theme
# ============================================================
def apply_visual_theme(theme_name: str) -> None:
    if theme_name == "Standard":
        return

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(1000px 500px at 95% -10%, rgba(92, 219, 149, 0.20), rgba(92, 219, 149, 0)),
                radial-gradient(900px 500px at 0% -20%, rgba(63, 132, 255, 0.20), rgba(63, 132, 255, 0)),
                linear-gradient(180deg, #0c1118 0%, #0d131b 100%);
        }
        .block-container { padding-top: 1.1rem; padding-bottom: 1rem; }
        h1, h2, h3 { letter-spacing: 0.01em; }
        [data-testid="stMetric"] {
            border: 1px solid rgba(106, 188, 138, 0.35);
            background: rgba(18, 24, 20, 0.35);
            border-radius: 12px;
            padding: 0.35rem 0.5rem;
        }
        [data-testid="stChatMessage"] {
            border-left: 2px solid rgba(106, 188, 138, 0.55);
            padding-left: 0.5rem;
        }
        [data-testid="stTabs"] button[role="tab"] { height: 2.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Prompting helpers
# ============================================================
def build_system_prompt(mode: str, custom_note: str) -> str:
    mode_rules = {
        "General Guide": "Give practical tabletop guidance with clean structure.",
        "World Builder": "Prioritize lore consistency, factions, cultures, geography, and history.",
        "Narrative Designer": "Focus on arcs, hooks, emotional beats, and session pacing.",
        "Encounter Architect": "Design balanced encounters with tactical options and clear stakes.",
    }
    base = (
        "You are a senior TTRPG co-designer helping build worlds and playable session content. "
        "Be concise by default and use markdown structure for reusable outputs."
    )
    return f"{base}\nMode directive: {mode_rules.get(mode,'World Builder')}\nCustom note: {custom_note or 'None'}"


def canon_context_from_world_bible(max_entries: int = 8, max_chars: int = 6000) -> str:
    """Create a compact canon block from recent entries.

    Future improvement:
    - Replace this with retrieval (vector search) + entity graph for long-term scaling.
    """
    if not st.session_state.world_bible:
        return ""

    entries = list(reversed(st.session_state.world_bible))[:max_entries]
    blocks: List[str] = []
    for e in entries:
        title = e.get("title", "Untitled")
        tags = ", ".join(e.get("tags", []) or [])
        content = e.get("content", "")
        block = f"### {title}\nTags: {tags or 'None'}\n{content}".strip()
        blocks.append(block)

    canon = "\n\n".join(blocks)
    canon = canon[:max_chars]
    return "\n\n# Canon Reference (World Bible)\n" + canon


# ============================================================
# Token estimation + spend meter
# ============================================================
def try_tiktoken_count(text: str, model: str) -> Optional[int]:
    """Best-effort token count. Returns None if tiktoken isn't installed."""
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None

    try:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return None


def rough_token_estimate(text: str) -> int:
    """Fallback token estimate if tiktoken isn't available."""
    # Rule of thumb: ~4 chars/token
    return max(1, int(math.ceil(len(text) / 4)))


def estimate_tokens_for_messages(messages: List[Dict[str, str]], model: str) -> int:
    joined = "\n".join([f"{m.get('role','')}: {m.get('content','')}" for m in messages])
    tok = try_tiktoken_count(joined, model)
    return tok if tok is not None else rough_token_estimate(joined)


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def compute_cost_usd(usage: Usage, prompt_usd_per_1m: float, completion_usd_per_1m: float) -> float:
    return (usage.prompt_tokens / 1_000_000) * prompt_usd_per_1m + (usage.completion_tokens / 1_000_000) * completion_usd_per_1m


def record_usage(model: str, usage: Usage, cost_usd: float) -> None:
    totals = st.session_state.usage_total
    totals["prompt_tokens"] += int(usage.prompt_tokens)
    totals["completion_tokens"] += int(usage.completion_tokens)
    totals["total_tokens"] += int(usage.total_tokens)
    totals["cost_usd"] += float(cost_usd)

    by_model = totals["by_model"]
    if model not in by_model:
        by_model[model] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    by_model[model]["prompt_tokens"] += int(usage.prompt_tokens)
    by_model[model]["completion_tokens"] += int(usage.completion_tokens)
    by_model[model]["total_tokens"] += int(usage.total_tokens)
    by_model[model]["cost_usd"] += float(cost_usd)


# ============================================================
# Provider compatibility (streaming + non-streaming)
# ============================================================
def client_for(api_key: str, base_url: str):
    """Create a client compatible with openai>=1.x."""
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def stream_chat_response(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> Iterable[str]:
    """Yield assistant chunks. (Usage is usually unavailable in streaming.)"""
    if hasattr(openai, "OpenAI"):
        c = client_for(api_key, base_url)
        stream = c.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            text = getattr(delta, "content", None) if delta else None
            if text:
                yield text
        return

    # Legacy SDK path (openai<1.x)
    openai.api_key = api_key
    openai.api_base = base_url
    stream = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        text = choices[0].get("delta", {}).get("content")
        if text:
            yield text


def complete_chat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> Tuple[str, Optional[Usage]]:
    """Non-streaming completion that may return usage; else we estimate."""
    if hasattr(openai, "OpenAI"):
        c = client_for(api_key, base_url)
        resp = c.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage_obj = getattr(resp, "usage", None)
        if usage_obj:
            pt = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage_obj, "completion_tokens", 0) or 0)
            return text, Usage(prompt_tokens=pt, completion_tokens=ct)
        return text, None

    openai.api_key = api_key
    openai.api_base = base_url
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    text = (resp["choices"][0]["message"]["content"] or "").strip()
    usage_dict = resp.get("usage")
    if usage_dict:
        return text, Usage(
            prompt_tokens=int(usage_dict.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_dict.get("completion_tokens", 0) or 0),
        )
    return text, None


# ============================================================
# Project Foundry helpers
# ============================================================
def compose_blueprint(data: Dict[str, Any]) -> str:
    return dedent(
        f"""
        # {data['project_name']} — Campaign Blueprint

        **System:** {data['game_system']}  
        **Play Style:** {data['play_style']}  
        **Campaign Scope:** {data['campaign_scope']} ({data['session_length']} hour sessions)  
        **Mood Dial:** Grit {data['grit']}/10 · Wonder {data['wonder']}/10 · Intrigue {data['intrigue']}/10

        ## Core Premise
        {data['premise']}

        ## World Pillars
        - Genre Blend: {', '.join(data['genres'])}
        - Major Themes: {', '.join(data['themes'])}
        - Threat Horizon: {data['threat_horizon']}
        - Magic / Tech Level: {data['power_level']}

        ## VTT Session Tooling
        - Preferred VTT: {data['vtt_platform']}
        - Map Style: {data['map_style']}
        - Fog of War: {'Enabled' if data['fog_of_war'] else 'Disabled'}
        - Dynamic Lighting: {'Enabled' if data['dynamic_lighting'] else 'Disabled'}
        - Safety Tools: {', '.join(data['safety_tools'])}

        ## Build Checklist
        - Starting Region: {data['starting_region']}
        - Signature Faction: {data['signature_faction']}
        - Session Zero Focus: {data['session_zero_focus']}
        - Hook Generator Bias: {data['hook_bias']}
        - NPC Density Dial: {data['npc_density']}/10
        - Exploration Density Dial: {data['exploration_density']}/10

        ## Starter Deliverables
        1. One-page player primer
        2. Three factions with motives and clocks
        3. Five location cards
        4. Ten NPC seeds with voice tags
        5. Session 1 cold open + two branch paths
        """
    ).strip()


def session1_generation_prompt(blueprint_md: str) -> List[Dict[str, str]]:
    sys = (
        "You are a veteran TTRPG campaign designer. "
        "Generate Session 1 materials that are immediately playable, table-facing, and VTT-ready. "
        "Keep it tight, structured, and actionable."
    )
    user = dedent(
        f"""
        Use this blueprint and the canon reference (if present) to generate Session 1.

        REQUIRED OUTPUT (Markdown):
        1) Cold open scene (boxed text + sensory cues + immediate choice)
        2) 3 scene beats with branch points (A/B) and consequences
        3) 6 NPCs: name, role, motive, secret, voice tag, what they want *tonight*
        4) 3 locations: description + 2 interactive details + 1 hidden detail
        5) 2 encounter frameworks: one social, one tactical (include difficulty notes and levers)
        6) Loot/Clues: 6 items or leads; each points to a faction or clock
        7) GM cheat sheet: clocks, faction moves, if-then escalation table

        BLUEPRINT:
        {blueprint_md}
        """
    ).strip()
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ============================================================
# World Bible -> Mini Knowledge Graph
# ============================================================
ENTITY_RE = re.compile(r"\[\[([^\]]+)\]\]")  # explicit links: [[Entity Name]]
PROPER_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")


def normalize_entity(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name


def extract_entities(entry: Dict[str, Any]) -> List[str]:
    """Extract entities from an entry.

    Heuristics:
    - Prefer explicit wiki-style links [[Entity Name]]
    - Add title as a node
    - Mine capitalized phrases from content (crude NER)
    - Add tags as tag:... nodes
    """
    title = (entry.get("title") or "").strip()
    content = (entry.get("content") or "").strip()
    tags = entry.get("tags") or []

    entities: List[str] = []
    if title:
        entities.append(normalize_entity(title))

    for m in ENTITY_RE.findall(content):
        entities.append(normalize_entity(m))

    for m in PROPER_PHRASE_RE.findall(content):
        cand = normalize_entity(m)
        if len(cand) < 3:
            continue
        if cand in {"You", "GM", "Session", "World", "Canon", "Tags"}:
            continue
        entities.append(cand)

    for t in tags:
        t = str(t).strip()
        if t:
            entities.append(f"tag:{t}")

    # de-dupe
    seen = set()
    out: List[str] = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def build_knowledge_graph() -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], int]]:
    """Build a simple undirected weighted co-occurrence graph from the World Bible."""
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[Tuple[str, str], int] = {}

    for entry in st.session_state.world_bible:
        ents = extract_entities(entry)

        for e in ents:
            if e not in nodes:
                node_type = "tag" if e.startswith("tag:") else "entity"
                nodes[e] = {"type": node_type, "count": 0}
            nodes[e]["count"] += 1

        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                a, b = ents[i], ents[j]
                if a == b:
                    continue
                key = tuple(sorted((a, b)))
                edges[key] = edges.get(key, 0) + 1

    return nodes, edges


def rebuild_kg_if_needed(force: bool = False) -> None:
    if force or st.session_state.kg_built_at is None:
        nodes, edges = build_knowledge_graph()
        st.session_state.kg_nodes = nodes
        st.session_state.kg_edges = edges
        st.session_state.kg_built_at = now_iso()
        log_event("Knowledge graph rebuilt")


def neighbors_of(node: str, edges: Dict[Tuple[str, str], int]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for (a, b), w in edges.items():
        if a == node:
            out.append((b, w))
        elif b == node:
            out.append((a, w))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def graphviz_subgraph(center: str, max_neighbors: int = 10) -> str:
    nodes = st.session_state.kg_nodes
    edges = st.session_state.kg_edges
    if center not in nodes:
        return 'graph G { label="No graph"; }'

    neigh = neighbors_of(center, edges)[:max_neighbors]
    lines = ["graph G {", "  rankdir=LR;", "  node [shape=box, style=rounded];"]
    center_label = center.replace('"', '\\"')
    lines.append(f'  "{center_label}" [shape=doubleoctagon];')

    for n, w in neigh:
        n_label = n.replace('"', '\\"')
        pen = clamp(1 + (w * 0.6), 1, 6)
        lines.append(f'  "{n_label}";')
        lines.append(f'  "{center_label}" -- "{n_label}" [penwidth={pen:.2f}, label="{w}"];')

    lines.append("}")
    return "\n".join(lines)


# ============================================================
# Sidebar (Command Deck)
# ============================================================
with st.sidebar:
    st.header("⚙️ Command Deck")

    visual_theme = st.selectbox("Visual theme", ["TechnoDruid 20%", "Standard"], index=0)
    apply_visual_theme(visual_theme)

    provider = st.selectbox("Provider", ["OpenAI", "OpenRouter"])

    if provider == "OpenAI":
        default_base_url = "https://api.openai.com/v1"
        default_model = "gpt-4o-mini"
        key_label = "OpenAI API Key"
    else:
        default_base_url = "https://openrouter.ai/api/v1"
        default_model = "openai/gpt-4o-mini"
        key_label = "OpenRouter API Key"

    api_key = st.text_input(key_label, type="password", help="Session-only; never persisted.")
    base_url = st.text_input("Base URL", value=default_base_url)

    if provider == "OpenRouter":
        model = st.selectbox("Model", OPENROUTER_MODEL_OPTIONS, index=0)
    else:
        model = st.text_input("Model", value=default_model)

    ttrpg_mode = st.selectbox(
        "Assistant mode",
        ["World Builder", "General Guide", "Narrative Designer", "Encounter Architect"],
        index=0,
    )
    temperature = st.slider("Creativity", min_value=0.0, max_value=1.4, value=0.7, step=0.1)

    st.divider()
    st.subheader("💰 Spend Meter (est.)")
    prompt_cost = st.number_input(
        "Prompt $ / 1M tokens",
        min_value=0.0,
        value=float(DEFAULT_COSTS_PER_1M["prompt_usd_per_1m"]),
        step=0.10,
        help="Set to your actual provider pricing if you want accurate spend.",
    )
    completion_cost = st.number_input(
        "Completion $ / 1M tokens",
        min_value=0.0,
        value=float(DEFAULT_COSTS_PER_1M["completion_usd_per_1m"]),
        step=0.10,
        help="Set to your actual provider pricing if you want accurate spend.",
    )

    totals = st.session_state.usage_total
    st.metric("Total tokens (est.)", f"{totals['total_tokens']:,}")
    st.metric("Estimated spend", f"${totals['cost_usd']:.4f}")

    c1, c2 = st.columns(2)
    if c1.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        log_event("Conversation cleared")
        st.rerun()

    if c2.button("♻️ Rebuild KG", use_container_width=True):
        rebuild_kg_if_needed(force=True)
        st.rerun()


# ============================================================
# Main Layout
# ============================================================
st.title("🧙 TTRPG Studio")
st.caption("Immersive worldbuilding workspace: low-scroll UI, faster creation loops, VTT-ready planning.")

custom_system_note = st.text_input(
    "Behavior note",
    value="",
    placeholder="Example: prioritize gothic political intrigue and concise bullet outputs.",
)
system_prompt = build_system_prompt(ttrpg_mode, custom_system_note)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Provider", provider)
m2.metric("Model", model)
m3.metric("Messages", len(st.session_state.messages))
m4.metric("World Entries", len(st.session_state.world_bible))
m5.metric("Blueprints", len(st.session_state.project_blueprints))

chat_tab, foundry_tab, world_tab, prompt_tab, admin_tab = st.tabs(
    ["💬 Chat", "🧱 Project Foundry", "📚 World Bible + KG", "🖼️ Prompt Forge", "🛠️ Control Room"]
)


# ============================================================
# Chat Tab
# ============================================================
with chat_tab:
    st.subheader("Campaign Co-Pilot")
    transcript = st.container(height=470, border=True)
    with transcript:
        if not st.session_state.messages:
            st.info("Kick off with a question: lore, factions, quest arcs, encounters, or session prep.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Ask for lore, encounters, factions, NPCs, or scene beats...")
    if prompt:
        if not api_key:
            st.warning("Add your API key in the sidebar before chatting.", icon="🗝️")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        log_event("User sent chat message")

        canon = canon_context_from_world_bible()
        request_messages = [{"role": "system", "content": system_prompt + canon}] + [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
        ]

        prompt_tokens_est = estimate_tokens_for_messages(request_messages, model)

        try:
            with transcript:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    t0 = time.time()
                    response = st.write_stream(
                        stream_chat_response(
                            api_key=api_key,
                            base_url=base_url,
                            model=model,
                            messages=request_messages,
                            temperature=temperature,
                        )
                    )
                    t1 = time.time()

            st.session_state.messages.append({"role": "assistant", "content": response})
            log_event(f"Assistant response streamed ({t1 - t0:.2f}s)")

            completion_tokens_est = try_tiktoken_count(response, model) or rough_token_estimate(response)
            usage = Usage(prompt_tokens=prompt_tokens_est, completion_tokens=completion_tokens_est)
            cost = compute_cost_usd(usage, prompt_cost, completion_cost)
            record_usage(model, usage, cost)

        except Exception as exc:  # noqa: BLE001
            log_event(f"Request failed: {exc}")
            st.error(f"Request failed: {exc}")


# ============================================================
# Project Foundry Tab
# ============================================================
with foundry_tab:
    st.subheader("Project Foundry — Build a TTRPG/VTT Project From Scratch")
    st.caption("Menus, selectables, dials, and toggles for rapid campaign bootstrapping.")

    a, b, c = st.columns(3)
    project_name = a.text_input("Project name", "Ashes of the Verdant Crown")
    game_system = b.selectbox("Game system", ["D&D 5e", "Pathfinder 2e", "Blades in the Dark", "Mothership", "Custom"])
    vtt_platform = c.selectbox("VTT platform", ["Foundry", "Roll20", "Owlbear Rodeo", "Fantasy Grounds", "Alchemy"])

    d, e, f = st.columns(3)
    play_style = d.selectbox("Play style", ["Narrative-first", "Balanced", "Tactical-heavy", "Sandbox exploration"])
    campaign_scope = e.selectbox("Campaign scope", ["One-shot", "Mini-campaign", "Long campaign", "Open table"])
    session_length = f.select_slider("Session length", options=[2, 3, 4, 5, 6], value=4)

    genres = st.multiselect(
        "Genre blend",
        ["High Fantasy", "Dark Fantasy", "Cosmic Horror", "Post-Apocalyptic", "Steampunk", "Mythic", "Political Intrigue"],
        default=["Dark Fantasy", "Political Intrigue"],
    )
    themes = st.multiselect(
        "Themes",
        ["Legacy", "Survival", "Corruption", "Rebellion", "Discovery", "Faith", "Forbidden Knowledge"],
        default=["Legacy", "Corruption", "Discovery"],
    )

    g1, g2, g3 = st.columns(3)
    grit = g1.slider("Grit dial", 0, 10, 7)
    wonder = g2.slider("Wonder dial", 0, 10, 5)
    intrigue = g3.slider("Intrigue dial", 0, 10, 8)

    h1, h2, h3 = st.columns(3)
    power_level = h1.selectbox("Magic/tech level", ["Low", "Medium", "High", "Unstable"])
    threat_horizon = h2.selectbox("Threat horizon", ["Local", "Regional", "Kingdom-scale", "World-ending"])
    hook_bias = h3.selectbox("Hook generator bias", ["Mystery", "Faction conflict", "Heists", "Wilderness peril", "Court politics"])

    i1, i2, i3 = st.columns(3)
    map_style = i1.selectbox("Map style", ["Hand-drawn parchment", "Painterly", "Grid tactical", "Isometric", "Theater of the mind hybrid"])
    npc_density = i2.slider("NPC density", 1, 10, 6)
    exploration_density = i3.slider("Exploration density", 1, 10, 7)

    j1, j2 = st.columns(2)
    safety_tools = j1.multiselect(
        "Safety tools",
        ["Lines & Veils", "X-Card", "Open Door", "Script Change", "Stars & Wishes"],
        default=["Lines & Veils", "Stars & Wishes"],
    )
    session_zero_focus = j2.selectbox(
        "Session zero focus",
        ["Party bonds", "Faction ties", "Shared tragedy", "Crew charter", "Hexcrawl goals"],
    )

    k1, k2 = st.columns(2)
    starting_region = k1.text_input("Starting region", "The Thornwake Marches")
    signature_faction = k2.text_input("Signature faction", "The Emerald Synod")
    premise = st.text_area(
        "Core premise",
        value="The old druidic engine beneath the marsh awakens, and every faction wants to claim its weather-shaping power.",
        height=90,
    )

    fog_of_war = st.toggle("Enable Fog of War", value=True)
    dynamic_lighting = st.toggle("Enable Dynamic Lighting", value=True)

    blueprint_data = {
        "project_name": project_name,
        "game_system": game_system,
        "vtt_platform": vtt_platform,
        "play_style": play_style,
        "campaign_scope": campaign_scope,
        "session_length": session_length,
        "genres": genres or ["Fantasy"],
        "themes": themes or ["Discovery"],
        "grit": grit,
        "wonder": wonder,
        "intrigue": intrigue,
        "power_level": power_level,
        "threat_horizon": threat_horizon,
        "hook_bias": hook_bias,
        "map_style": map_style,
        "npc_density": npc_density,
        "exploration_density": exploration_density,
        "safety_tools": safety_tools or ["Open Door"],
        "session_zero_focus": session_zero_focus,
        "starting_region": starting_region,
        "signature_faction": signature_faction,
        "premise": premise,
        "fog_of_war": fog_of_war,
        "dynamic_lighting": dynamic_lighting,
    }
    blueprint_md = compose_blueprint(blueprint_data)

    st.markdown("#### Live Blueprint Preview")
    st.markdown(blueprint_md)

    x1, x2, x3 = st.columns(3)

    if x1.button("📥 Save blueprint", use_container_width=True):
        st.session_state.project_blueprints.append(
            {
                "name": project_name,
                "created_at": now_iso(),
                "blueprint": blueprint_md,
            }
        )
        log_event(f"Project blueprint saved: {project_name}")
        st.success("Blueprint saved to project vault.")

    if x2.button("📚 Add blueprint to World Bible", use_container_width=True):
        st.session_state.world_bible.append(
            {
                "title": f"Campaign Blueprint: {project_name}",
                "tags": ["blueprint", "project-foundry", game_system, vtt_platform],
                "content": blueprint_md,
                "created_at": now_iso(),
            }
        )
        rebuild_kg_if_needed(force=True)
        log_event(f"Blueprint pushed to world bible: {project_name}")
        st.success("Blueprint added to World Bible.")

    if x3.button("⚡ Generate Session 1", use_container_width=True):
        if not api_key:
            st.warning("Add your API key in the sidebar before generating.", icon="🗝️")
            st.stop()

        canon = canon_context_from_world_bible()
        msgs = session1_generation_prompt(blueprint_md)
        msgs[0]["content"] = msgs[0]["content"] + canon  # inject canon into system

        pt_est = estimate_tokens_for_messages(msgs, model)

        with st.spinner("Generating Session 1..."):
            try:
                text, usage = complete_chat(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    messages=msgs,
                    temperature=clamp(temperature, 0.2, 1.2),
                )
                if usage is None:
                    ct_est = try_tiktoken_count(text, model) or rough_token_estimate(text)
                    usage = Usage(prompt_tokens=pt_est, completion_tokens=ct_est)

                cost = compute_cost_usd(usage, prompt_cost, completion_cost)
                record_usage(model, usage, cost)

                st.session_state.world_bible.append(
                    {
                        "title": f"Session 1 Pack: {project_name}",
                        "tags": ["session-1", "generated", "project-foundry", game_system, vtt_platform],
                        "content": text,
                        "created_at": now_iso(),
                    }
                )
                rebuild_kg_if_needed(force=True)
                log_event(f"Session 1 generated and saved to World Bible: {project_name}")
                st.success("Session 1 generated and added to World Bible.")
                st.markdown("#### Session 1 Output")
                st.markdown(text)
            except Exception as exc:  # noqa: BLE001
                log_event(f"Session 1 generation failed: {exc}")
                st.error(f"Generation failed: {exc}")


# ============================================================
# World Bible + Knowledge Graph Tab
# ============================================================
with world_tab:
    st.subheader("World Bible + Knowledge Graph")
    st.caption("Canon notes + a lightweight entity graph for fast recall and continuity checks.")

    left, right = st.columns([1, 2])

    with left:
        with st.form("world_bible_form", clear_on_submit=True):
            entry_title = st.text_input("Entry title")
            entry_tags = st.text_input("Tags (comma-separated)")
            entry_content = st.text_area(
                "Lore content",
                height=180,
                help="Tip: Use [[Entity Name]] to explicitly link entities for the knowledge graph.",
            )
            add_entry = st.form_submit_button("Add entry")

        if add_entry:
            if entry_title.strip() and entry_content.strip():
                st.session_state.world_bible.append(
                    {
                        "title": entry_title.strip(),
                        "tags": [t.strip() for t in entry_tags.split(",") if t.strip()],
                        "content": entry_content.strip(),
                        "created_at": now_iso(),
                    }
                )
                rebuild_kg_if_needed(force=True)
                log_event(f"World entry added: {entry_title.strip()}")
                st.success("World entry added.")
            else:
                st.warning("Please provide an entry title and lore content.")

        st.divider()
        st.markdown("### 🔎 Search")
        rebuild_kg_if_needed(force=False)

        query = st.text_input("Search World Bible / Entities", placeholder="e.g., Emerald Synod, Thornwake, tag:blueprint")
        search_mode = st.radio("Search target", ["Entries", "Entities"], horizontal=True, index=0)

        if query:
            q = query.strip().lower()
            if search_mode == "Entries":
                matches = []
                for e in reversed(st.session_state.world_bible):
                    hay = (e.get("title", "") + "\n" + e.get("content", "") + "\n" + ",".join(e.get("tags", []) or [])).lower()
                    if q in hay:
                        matches.append(e)
                st.caption(f"Matches: {len(matches)}")
                for i, e in enumerate(matches[:20], start=1):
                    with st.expander(f"{i}. {e.get('title','Untitled')}"):
                        st.write(f"**Tags:** {', '.join(e.get('tags', []) or []) or 'None'}")
                        st.write(e.get("content", ""))
                        st.caption(f"Created: {e.get('created_at','')}")
            else:
                nodes = st.session_state.kg_nodes
                hits = [n for n in nodes.keys() if q in n.lower()]
                st.caption(f"Entity matches: {len(hits)}")
                for n in hits[:30]:
                    st.write(f"- **{n}** · type={nodes[n]['type']} · mentions={nodes[n]['count']}")

    with right:
        entries_container = st.container(height=330, border=True)
        with entries_container:
            if not st.session_state.world_bible:
                st.info("No lore entries yet. Add your first canon note on the left.")
            else:
                for idx, entry in enumerate(reversed(st.session_state.world_bible)):
                    with st.expander(f"{len(st.session_state.world_bible)-idx}. {entry.get('title','Untitled')}"):
                        st.write(f"**Tags:** {', '.join(entry.get('tags', []) or []) or 'None'}")
                        st.write(entry.get("content", ""))

        st.markdown("### 🕸️ Knowledge Graph (local view)")
        nodes = st.session_state.kg_nodes
        edges = st.session_state.kg_edges

        if not nodes:
            st.info("Knowledge graph is empty. Add World Bible entries to populate it.")
        else:
            popular = sorted(nodes.items(), key=lambda kv: kv[1]["count"], reverse=True)[:50]
            center = st.selectbox(
                "Center entity",
                options=[n for n, _ in popular] + [n for n in nodes.keys() if n not in dict(popular)],
                index=0,
                help="Tip: tags are shown as tag:... nodes.",
            )
            max_neighbors = st.slider("Max neighbors", 3, 20, 10)

            neigh = neighbors_of(center, edges)[:max_neighbors]
            if neigh:
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Top connections**")
                    for n, w in neigh:
                        st.write(f"- {n} ({w})")
                with cols[1]:
                    st.markdown("**Graph snapshot**")
                    st.graphviz_chart(graphviz_subgraph(center, max_neighbors=max_neighbors), use_container_width=True)
            else:
                st.info("No neighbors for this node yet. Add more entries or use [[Entity]] links.")

            st.caption(f"KG built: {st.session_state.kg_built_at or 'never'}")


# ============================================================
# Prompt Forge Tab
# ============================================================
with prompt_tab:
    st.subheader("Prompt Forge")

    col_a, col_b, col_c = st.columns(3)
    subject = col_a.text_input("Subject", "Ancient druid citadel over flooded ruins")
    style = col_b.selectbox("Style", ["Cinematic fantasy", "Oil painting", "Dark matte concept", "Illustrative map-art"])
    lighting = col_c.selectbox("Lighting", ["Moonlit bioluminescence", "Golden hour haze", "Stormlight flashes", "Torchlit gloom"])

    col_d, col_e, col_f = st.columns(3)
    mood = col_d.selectbox("Mood", ["Sacred", "Threatening", "Melancholic", "Awe-inspiring"])
    camera = col_e.selectbox("Framing", ["Wide establishing", "Character close-up", "Bird's-eye tactical", "Isometric angle"])
    detail_dial = col_f.slider("Detail dial", 1, 10, 8)

    negative = st.text_input("Negative prompt", "blurry, low detail, text artifacts, watermark")
    generated_prompt = (
        f"{subject}. Style: {style}. Lighting: {lighting}. Mood: {mood}. "
        f"Framing: {camera}. Detail level {detail_dial}/10. Negative prompt: {negative}."
    )

    st.code(generated_prompt, language="text")
    if st.button("Save prompt to vault"):
        st.session_state.prompt_vault.append(
            {
                "prompt": generated_prompt,
                "subject": subject,
                "style": style,
                "created_at": now_iso(),
            }
        )
        log_event("Image prompt saved to vault")
        st.success("Prompt saved.")

    st.markdown("#### Vault")
    if not st.session_state.prompt_vault:
        st.info("No prompts saved yet.")
    else:
        for idx, item in enumerate(reversed(st.session_state.prompt_vault)):
            with st.expander(f"{len(st.session_state.prompt_vault)-idx}. {item.get('subject') or 'Untitled'}"):
                st.code(item["prompt"], language="text")
                st.caption(f"Created: {item.get('created_at','')}")


# ============================================================
# Control Room / Admin Tab
# ============================================================
with admin_tab:
    st.subheader("Control Room")

    payload = {
        "messages": st.session_state.messages,
        "world_bible": st.session_state.world_bible,
        "prompt_vault": st.session_state.prompt_vault,
        "project_blueprints": st.session_state.project_blueprints,
        "audit_log": st.session_state.audit_log,
        "usage_total": st.session_state.usage_total,
    }

    t1, t2 = st.columns(2)
    t1.download_button(
        "Export workspace JSON",
        data=json.dumps(payload, indent=2),
        file_name="ttrpg_studio_export.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded = t2.file_uploader("Import workspace JSON", type=["json"])
    if uploaded is not None:
        try:
            imported = json.loads(uploaded.read().decode("utf-8"))
            for key in ["messages", "world_bible", "prompt_vault", "project_blueprints", "audit_log"]:
                if key in imported and isinstance(imported[key], list):
                    st.session_state[key] = imported[key]
            if "usage_total" in imported and isinstance(imported["usage_total"], dict):
                st.session_state.usage_total = imported["usage_total"]
            rebuild_kg_if_needed(force=True)
            st.success("Workspace imported.")
            log_event("Workspace imported from JSON")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Import failed: {exc}")

    st.markdown("#### Spend Meter Breakdown")
    totals = st.session_state.usage_total
    st.write(
        f"- Prompt tokens: **{totals['prompt_tokens']:,}**\n"
        f"- Completion tokens: **{totals['completion_tokens']:,}**\n"
        f"- Total tokens: **{totals['total_tokens']:,}**\n"
        f"- Estimated spend: **${totals['cost_usd']:.4f}**"
    )
    if totals.get("by_model"):
        st.markdown("**By model**")
        for m, d in sorted(totals["by_model"].items(), key=lambda kv: kv[1]["total_tokens"], reverse=True):
            st.write(f"- `{m}` — {d['total_tokens']:,} tokens — ${d['cost_usd']:.4f}")

    st.markdown("#### Saved Blueprints")
    if not st.session_state.project_blueprints:
        st.info("No saved blueprints yet.")
    else:
        for idx, blueprint in enumerate(reversed(st.session_state.project_blueprints)):
            with st.expander(f"{len(st.session_state.project_blueprints)-idx}. {blueprint['name']}"):
                st.markdown(blueprint["blueprint"])

    st.markdown("#### Audit Log")
    if not st.session_state.audit_log:
        st.info("No events logged yet.")
    else:
        for line in reversed(st.session_state.audit_log[-60:]):
            st.write(f"- {line}")
