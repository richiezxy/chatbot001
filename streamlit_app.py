"""TTRPG worldbuilding chat studio.

This Streamlit app is intentionally structured and heavily commented so it can evolve
from a single-file prototype into a larger "t3.chat style" daily-use workspace.

Current goals implemented:
- Multi-provider chat (OpenAI / OpenRouter)
- TTRPG-oriented system prompts and world bible notes
- Image prompt builder + vault storage (generation can be added later)
- Admin-style controls: audit log, JSON export/import, session insights
- A subtle "20% TechnoDruid" visual theme option
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import streamlit as st
import openai

# -------------------------------
# App bootstrap and base settings
# -------------------------------
st.set_page_config(page_title="TTRPG Studio", page_icon="🧙", layout="wide")


# -------------------------------
# Session state initialization
# -------------------------------
def ensure_state() -> None:
    """Initialize all session-level stores used by the app.

    Keeping state keys centralized makes the app easier to maintain and lets us
    later migrate storage from in-memory to DB-backed persistence.
    """

    defaults: dict[str, Any] = {
        "messages": [],
        "world_bible": [],
        "prompt_vault": [],
        "audit_log": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


ensure_state()


# -------------------------------
# Theming helpers
# -------------------------------
def apply_visual_theme(theme_name: str) -> None:
    """Apply lightweight app styling.

    "TechnoDruid 20%" means subtle futuristic + natural accents,
    not a full neon overhaul.
    """

    if theme_name != "TechnoDruid 20%":
        return

    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top right, #112118 0%, #0f1117 45%, #0b0d12 100%);
        }
        .stMetric {
            border: 1px solid rgba(106, 188, 138, 0.35);
            border-radius: 12px;
            padding: 8px;
            background: rgba(18, 24, 20, 0.35);
        }
        .stChatMessage {
            border-left: 2px solid rgba(106, 188, 138, 0.55);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------
# Sidebar configuration
# -------------------------------
with st.sidebar:
    st.header("⚙️ Workspace Settings")

    visual_theme = st.selectbox("Visual theme", ["Standard", "TechnoDruid 20%"], index=1)
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

    api_key = st.text_input(key_label, type="password")
    base_url = st.text_input("Base URL", value=default_base_url)
    model = st.text_input("Model", value=default_model)

    ttrpg_mode = st.selectbox(
        "TTRPG assistant mode",
        ["General Guide", "World Builder", "Narrative Designer", "Encounter Architect"],
        index=1,
    )

from openai import OpenAI

st.set_page_config(page_title="Multi-Provider Chat", page_icon="💬", layout="wide")

st.title("💬 AI Chat Studio")
st.caption(
    "Chat with either OpenAI or OpenRouter models. Configure provider, model, and behavior from the sidebar."
)

with st.sidebar:
    st.header("⚙️ Chat Settings")

    provider = st.selectbox("Provider", ["OpenAI", "OpenRouter"])

    if provider == "OpenAI":
        default_base_url = "https://api.openai.com/v1"
        default_model = "gpt-4o-mini"
        key_label = "OpenAI API Key"
        model_help = "Any OpenAI chat/completions model you have access to."
    else:
        default_base_url = "https://openrouter.ai/api/v1"
        default_model = "openai/gpt-4o-mini"
        key_label = "OpenRouter API Key"
        model_help = "Use OpenRouter model IDs (e.g. anthropic/claude-3.5-sonnet)."

    api_key = st.text_input(key_label, type="password", help="Your key is only used for this session.")
    base_url = st.text_input("Base URL", value=default_base_url)
    model = st.text_input("Model", value=default_model, help=model_help)

    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
        help="Set assistant behavior for all replies.",
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.audit_log.append("Conversation cleared")
        st.rerun()


# -------------------------------
# Prompt composition helpers
# -------------------------------
def build_system_prompt(mode: str, custom_note: str) -> str:
    """Build a structured system prompt tuned to TTRPG use."""

    mode_rules = {
        "General Guide": "Give clear, practical guidance for tabletop storytelling.",
        "World Builder": "Prioritize lore consistency, factions, cultures, and geography.",
        "Narrative Designer": "Focus on arcs, hooks, tone, and session pacing.",
        "Encounter Architect": "Design balanced encounters, stakes, and tactical options.",
    }

    base = (
        "You are a senior TTRPG co-designer helping build worlds and playable content. "
        "Keep outputs structured, reusable, and concise unless asked for depth."
    )
    return f"{base}\nMode directive: {mode_rules[mode]}\nCustom note: {custom_note or 'None'}"


# -------------------------------
# Provider compatibility helpers
# -------------------------------
def stream_chat_response(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
):
    """Yield assistant text chunks from either modern or legacy OpenAI SDKs.

    Why this exists:
    - `openai>=1.x` exposes `openai.OpenAI(...)` client objects
    - many existing environments still use `openai<1.0` with module-level APIs

    This bridge prevents import/runtime failures like:
    `ImportError: cannot import name 'OpenAI' from 'openai'`.
    """

    # Modern SDK path (openai>=1.x)
    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        stream = client.chat.completions.create(
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
        delta = choices[0].get("delta", {})
        text = delta.get("content")
        if text:
            yield text



# -------------------------------
# Top-level app layout
# -------------------------------
st.title("🧙 TTRPG Studio")
st.caption("A daily-use worldbuilding workspace with chat, prompt vaults, and admin controls.")

custom_system_note = st.text_area(
    "Custom system note",
    value="",
    placeholder="Example: Keep lore dark-fantasy, avoid steampunk, favor political intrigue.",
)

system_prompt = build_system_prompt(ttrpg_mode, custom_system_note)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Provider", provider)
m2.metric("Model", model)
m3.metric("Chat messages", len(st.session_state.messages))
m4.metric("World entries", len(st.session_state.world_bible))

chat_tab, world_tab, prompt_tab, admin_tab, roadmap_tab = st.tabs(
    ["💬 Chat", "📚 World Bible", "🖼️ Prompt Forge", "🛠️ Admin", "🗺️ Roadmap"]
)


# -------------------------------
# Chat tab
# -------------------------------
with chat_tab:
    st.subheader("Conversation")
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Header metrics
c1, c2, c3 = st.columns(3)
c1.metric("Provider", provider)
c2.metric("Model", model)
c3.metric("Messages", len(st.session_state.messages))

chat_panel = st.container(border=True)
with chat_panel:
    st.subheader("Conversation")
    if not st.session_state.messages:
        st.info("Start by asking a question in the chat box below.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask for lore, encounters, factions, locations, NPCs...")
    if prompt:
        if not api_key:
            st.warning("Add your API key in the sidebar before chatting.", icon="🗝️")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.audit_log.append("User sent chat message")

        with st.chat_message("user"):
            st.markdown(prompt)

        request_messages = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
        ]

        try:
            with st.chat_message("assistant"):
                response = st.write_stream(
                    stream_chat_response(
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        messages=request_messages,
                        temperature=temperature,
                    )
                )

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.audit_log.append("Assistant response streamed")
        except Exception as exc:  # noqa: BLE001
            st.session_state.audit_log.append(f"Request failed: {exc}")
            st.error(f"Request failed: {exc}")


# -------------------------------
# World Bible tab
# -------------------------------
with world_tab:
    st.subheader("World Bible")
    st.caption("Store canonical lore snippets to keep continuity stable over long campaigns.")

    with st.form("world_bible_form", clear_on_submit=True):
        entry_title = st.text_input("Entry title")
        entry_tags = st.text_input("Tags (comma-separated)")
        entry_content = st.text_area("Lore content", height=160)
        add_entry = st.form_submit_button("Add entry")

    if add_entry and entry_title and entry_content:
        st.session_state.world_bible.append(
            {
                "title": entry_title.strip(),
                "tags": [tag.strip() for tag in entry_tags.split(",") if tag.strip()],
                "content": entry_content.strip(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        st.session_state.audit_log.append(f"World entry added: {entry_title.strip()}")
        st.success("World entry added.")

    for idx, entry in enumerate(st.session_state.world_bible):
        with st.expander(f"{idx + 1}. {entry['title']}"):
            st.write(f"**Tags:** {', '.join(entry['tags']) if entry['tags'] else 'None'}")
            st.write(entry["content"])


# -------------------------------
# Prompt Forge tab
# -------------------------------
with prompt_tab:
    st.subheader("Image Prompt Forge")
    st.caption("Create and store image prompts now; plug in generation providers later.")

    col_a, col_b = st.columns(2)
    with col_a:
        subject = st.text_input("Subject", placeholder="Ancient druid citadel on floating islands")
        style = st.text_input("Style", placeholder="cinematic fantasy concept art")
    with col_b:
        lighting = st.text_input("Lighting", placeholder="volumetric moonlight with bioluminescent fog")
        mood = st.text_input("Mood", placeholder="mysterious and sacred")

    camera = st.text_input("Camera / framing", placeholder="wide establishing shot, high contrast")
    negative = st.text_input("Negative prompt", placeholder="blurry, low detail, text artifacts")

    generated_prompt = (
        f"{subject}, {style}, {lighting}, mood: {mood}, camera: {camera}. "
        f"Negative prompt: {negative}."
    )
    st.code(generated_prompt, language="text")

    c_save, c_copy = st.columns([1, 3])
    if c_save.button("Save to vault"):
        st.session_state.prompt_vault.append(
            {
                "prompt": generated_prompt,
                "subject": subject,
                "style": style,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
if prompt := st.chat_input("Ask anything..."):
    if not api_key:
        st.warning("Please add an API key in the sidebar before chatting.", icon="🗝️")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = OpenAI(api_key=api_key, base_url=base_url)
    request_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=request_messages,
            temperature=temperature,
            stream=True,
        )
        st.session_state.audit_log.append("Image prompt saved to vault")
        st.success("Prompt saved.")

    st.markdown("#### Prompt Vault")
    for idx, item in enumerate(st.session_state.prompt_vault):
        with st.expander(f"{idx + 1}. {item.get('subject') or 'Untitled prompt'}"):
            st.code(item["prompt"], language="text")


# -------------------------------
# Admin tab
# -------------------------------
with admin_tab:
    st.subheader("Admin Controls")
    st.caption("Operational tools for managing session data and project governance.")

    payload = {
        "messages": st.session_state.messages,
        "world_bible": st.session_state.world_bible,
        "prompt_vault": st.session_state.prompt_vault,
        "audit_log": st.session_state.audit_log,
    }

    st.download_button(
        "Export workspace JSON",
        data=json.dumps(payload, indent=2),
        file_name="ttrpg_studio_export.json",
        mime="application/json",
    )

    uploaded = st.file_uploader("Import workspace JSON", type=["json"])
    if uploaded is not None:
        try:
            imported = json.loads(uploaded.read().decode("utf-8"))
            for key in ["messages", "world_bible", "prompt_vault", "audit_log"]:
                if key in imported and isinstance(imported[key], list):
                    st.session_state[key] = imported[key]
            st.success("Workspace imported.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Import failed: {exc}")

    st.markdown("#### Audit Log")
    if not st.session_state.audit_log:
        st.info("No events logged yet.")
    else:
        for line in reversed(st.session_state.audit_log[-20:]):
            st.write(f"- {line}")


# -------------------------------
# Roadmap tab (in-app visibility)
# -------------------------------
with roadmap_tab:
    st.subheader("Forward Roadmap")
    st.markdown(
        """
1. **Persistence layer**: move session data to SQLite/Postgres with per-user workspaces.
2. **Auth + roles**: admin/editor/viewer permissions and project-level access control.
3. **Image generation adapters**: connect OpenRouter/OpenAI/other image providers.
4. **TTRPG knowledge graph**: entities, factions, events, timelines, and canon conflict checks.
5. **Daily workflow tooling**: reminders, campaign TODOs, and reusable session templates.
6. **Observability**: token spend, response latency, and prompt quality analytics.
"""
    )
        with st.chat_message("assistant"):
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as exc:  # noqa: BLE001
        st.error(f"Request failed: {exc}")
