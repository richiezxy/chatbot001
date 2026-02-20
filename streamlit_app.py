"""Immersive TTRPG studio built with Streamlit.

Focused on fast, low-scroll workflows:
- Compact command-center UI
- Multi-provider chat
- World bible + prompt vault
- Project Foundry tab for building VTT-ready campaigns from scratch
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any

import openai
import streamlit as st

st.set_page_config(page_title="TTRPG Studio", page_icon="🧙", layout="wide")


def ensure_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "world_bible": [],
        "prompt_vault": [],
        "audit_log": [],
        "project_blueprints": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


ensure_state()


def log_event(event: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    st.session_state.audit_log.append(f"[{timestamp}] {event}")


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
    return f"{base}\nMode directive: {mode_rules[mode]}\nCustom note: {custom_note or 'None'}"


def stream_chat_response(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
):
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


def compose_blueprint(data: dict[str, Any]) -> str:
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
    model = st.text_input("Model", value=default_model)

    ttrpg_mode = st.selectbox(
        "Assistant mode",
        ["World Builder", "General Guide", "Narrative Designer", "Encounter Architect"],
        index=0,
    )
    temperature = st.slider("Creativity", min_value=0.0, max_value=1.4, value=0.7, step=0.1)

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        log_event("Conversation cleared")
        st.rerun()


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
    ["💬 Chat", "🧱 Project Foundry", "📚 World Bible", "🖼️ Prompt Forge", "🛠️ Control Room"]
)

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
            log_event("Assistant response streamed")
        except Exception as exc:  # noqa: BLE001
            log_event(f"Request failed: {exc}")
            st.error(f"Request failed: {exc}")

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

    x1, x2 = st.columns(2)
    if x1.button("📥 Save blueprint", use_container_width=True):
        st.session_state.project_blueprints.append(
            {
                "name": project_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
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
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        log_event(f"Blueprint pushed to world bible: {project_name}")
        st.success("Blueprint added to World Bible.")

with world_tab:
    st.subheader("World Bible")
    left, right = st.columns([1, 2])
    with left:
        with st.form("world_bible_form", clear_on_submit=True):
            entry_title = st.text_input("Entry title")
            entry_tags = st.text_input("Tags (comma-separated)")
            entry_content = st.text_area("Lore content", height=180)
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
            log_event(f"World entry added: {entry_title.strip()}")
            st.success("World entry added.")

    with right:
        entries_container = st.container(height=500, border=True)
        with entries_container:
            if not st.session_state.world_bible:
                st.info("No lore entries yet. Add your first canon note on the left.")
            for idx, entry in enumerate(reversed(st.session_state.world_bible)):
                with st.expander(f"{len(st.session_state.world_bible)-idx}. {entry['title']}"):
                    st.write(f"**Tags:** {', '.join(entry['tags']) if entry['tags'] else 'None'}")
                    st.write(entry["content"])

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
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        log_event("Image prompt saved to vault")
        st.success("Prompt saved.")

    st.markdown("#### Vault")
    for idx, item in enumerate(reversed(st.session_state.prompt_vault)):
        with st.expander(f"{len(st.session_state.prompt_vault)-idx}. {item.get('subject') or 'Untitled'}"):
            st.code(item["prompt"], language="text")

with admin_tab:
    st.subheader("Control Room")
    payload = {
        "messages": st.session_state.messages,
        "world_bible": st.session_state.world_bible,
        "prompt_vault": st.session_state.prompt_vault,
        "project_blueprints": st.session_state.project_blueprints,
        "audit_log": st.session_state.audit_log,
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
            st.success("Workspace imported.")
            log_event("Workspace imported from JSON")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Import failed: {exc}")

    st.markdown("#### Saved Blueprints")
    for idx, blueprint in enumerate(reversed(st.session_state.project_blueprints)):
        with st.expander(f"{len(st.session_state.project_blueprints)-idx}. {blueprint['name']}"):
            st.markdown(blueprint["blueprint"])

    st.markdown("#### Audit Log")
    if not st.session_state.audit_log:
        st.info("No events logged yet.")
    else:
        for line in reversed(st.session_state.audit_log[-40:]):
            st.write(f"- {line}")
