"""Immersive TTRPG studio built with Streamlit.

Focused on fast, low-scroll workflows:
- Compact command-center UI
- Multi-provider chat
- World bible + prompt vault
- Project Foundry tab for building Foundry-ready campaigns from scratch
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any

import openai
import streamlit as st

st.set_page_config(page_title="TTRPG Studio", page_icon="🧙", layout="wide")

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


# -------------------------------
# Session bootstrap
# -------------------------------
def ensure_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "world_bible": [],
        "prompt_vault": [],
        "audit_log": [],
        "project_blueprints": [],
        "lock_chat": False,
        "lock_foundry": False,
        "lock_world": False,
        "lock_prompt": False,
        "lock_admin": False,
        "foundry_blueprint": "",
        "world_entry_draft": "",
        "prompt_preview": "",
        "admin_summary": "",
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
                radial-gradient(1000px 500px at 90% -10%, rgba(102, 37, 219, 0.23), rgba(102, 37, 219, 0)),
                radial-gradient(900px 500px at 0% -20%, rgba(11, 147, 193, 0.20), rgba(11, 147, 193, 0)),
                linear-gradient(180deg, #0a0c12 0%, #10131d 100%);
        }
        .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 95rem; }
        h1, h2, h3 { letter-spacing: 0.02em; }
        [data-testid="stMetric"] {
            border: 1px solid rgba(162, 129, 255, 0.45);
            background: rgba(24, 18, 40, 0.42);
            border-radius: 12px;
            padding: 0.35rem 0.5rem;
        }
        [data-testid="stChatMessage"] {
            border-left: 2px solid rgba(119, 228, 255, 0.60);
            background: rgba(16, 19, 29, 0.40);
            border-radius: 6px;
            padding-left: 0.6rem;
            margin-bottom: 0.45rem;
        }
        [data-testid="stTabs"] button[role="tab"] {
            height: 2.1rem;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_system_prompt(mode: str, custom_note: str) -> str:
    mode_rules = {
        "World Builder": "Prioritize lore consistency, factions, corp politics, magical paradigms, and city texture.",
        "General Guide": "Give practical tabletop guidance with concise, reusable structure.",
        "Narrative Designer": "Focus on arcs, hooks, emotional beats, and session pacing.",
        "Encounter Architect": "Design balanced encounters with tactical options and clear stakes.",
    }
    base = (
        "You are a veteran GM co-designer for Foundry VTT campaigns. "
        "Treat Shadowrun 5e and Mage: The Ascension as core inspirations, but do not force a crossover unless asked. "
        "Blend cyberpunk conspiracies, occult mystery, and player-facing clarity. "
        "Use markdown headings and bullet points for outputs."
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
        # {data['project_name']} — Foundry Campaign Blueprint

        **Core Game:** {data['game_system']}  
        **Play Blend:** {data['play_style']}  
        **Campaign Scope:** {data['campaign_scope']} ({data['session_length']}h sessions)  
        **Tone Dials:** Noir {data['noir']}/10 · Horror {data['horror']}/10 · Hope {data['hope']}/10

        ## Core Premise
        {data['premise']}

        ## Setting DNA
        - City Layer: {data['city_layer']}
        - Paradigm Pressure: {data['paradigm_pressure']}
        - Corporate Temperature: {data['corp_heat']}
        - Metaplot Thread: {data['metaplot_thread']}

        ## Foundry VTT Stack
        - Platform: Foundry VTT
        - Modules Focus: {', '.join(data['modules_focus'])}
        - Scene Style: {data['scene_style']}
        - Fog of War: {'Enabled' if data['fog_of_war'] else 'Disabled'}
        - Dynamic Lighting: {'Enabled' if data['dynamic_lighting'] else 'Disabled'}
        - Safety Tools: {', '.join(data['safety_tools'])}

        ## Session Build Checklist
        - Opening District: {data['opening_district']}
        - Signature Antagonist: {data['signature_antagonist']}
        - Session Zero Focus: {data['session_zero_focus']}
        - Secondary Inspiration: {data['secondary_inspiration']}
        - Hook Bias: {data['hook_bias']}
        - Legwork Density: {data['legwork_density']}/10
        - Combat Density: {data['combat_density']}/10

        ## Starter Deliverables
        1. One-page player onboarding brief
        2. Three factions with clocks + leverage
        3. Five scene cards (street, astral, corporate, sanctum, fallout)
        4. Ten NPC seeds with voice tags + tells
        5. Session 1 opener with two branch vectors
        """
    ).strip()


# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Command Deck")
    visual_theme = st.selectbox("Visual theme", ["Neon Arcana", "Standard"], index=0)
    apply_visual_theme(visual_theme)

    provider = st.selectbox("Provider", ["OpenAI", "OpenRouter"])
    if provider == "OpenAI":
        default_base_url = "https://api.openai.com/v1"
        default_model = "gpt-4o-mini"
        key_label = "OpenAI API Key"
        model = st.text_input("Model", value=default_model)
    else:
        default_base_url = "https://openrouter.ai/api/v1"
        default_model = "openai/gpt-4o-mini"
        key_label = "OpenRouter API Key"
        selected_model = st.selectbox(
            "Model",
            OPENROUTER_MODEL_OPTIONS,
            index=OPENROUTER_MODEL_OPTIONS.index(default_model),
            help="Pick a preset model or override with a custom ID below.",
        )
        custom_model = st.text_input("Custom model override (optional)", value="")
        model = custom_model.strip() or selected_model

    api_key = st.text_input(key_label, type="password", help="Session-only; never persisted.")
    base_url = st.text_input("Base URL", value=default_base_url)

    ttrpg_mode = st.selectbox(
        "Assistant mode",
        ["World Builder", "Narrative Designer", "Encounter Architect", "General Guide"],
        index=0,
    )
    temperature = st.slider("Creativity", min_value=0.0, max_value=1.4, value=0.8, step=0.1)

    if st.button("🧹 Clear chat", use_container_width=True, disabled=st.session_state.lock_chat):
        st.session_state.messages = []
        log_event("Conversation cleared")
        st.rerun()


# -------------------------------
# Header
# -------------------------------
st.title("🧙 TTRPG Studio")
st.caption("Foundry-first campaign studio with Shadowrun 5e and Mage: The Ascension as core inspirations.")

custom_system_note = st.text_input(
    "Behavior note",
    value="",
    placeholder="Example: emphasize corp intrigue, paradox costs, and street-level consequences.",
)
system_prompt = build_system_prompt(ttrpg_mode, custom_system_note)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Core Inspirations", "SR5e / M:tA")
m2.metric("VTT", "Foundry")
m3.metric("Messages", len(st.session_state.messages))
m4.metric("World Entries", len(st.session_state.world_bible))
m5.metric("Blueprints", len(st.session_state.project_blueprints))

chat_tab, foundry_tab, world_tab, prompt_tab, admin_tab = st.tabs(
    ["💬 Chat", "🧱 Foundry Project", "📚 World Bible", "🖼️ Prompt Forge", "🛠️ Control Room"]
)


# -------------------------------
# Chat
# -------------------------------
with chat_tab:
    st.subheader("Campaign Co-Pilot")
    st.session_state.lock_chat = st.toggle("🔒 Lock chat", value=st.session_state.lock_chat)

    transcript = st.container(height=520, border=True)
    with transcript:
        if not st.session_state.messages:
            st.info("Ask for runs, chantry plots, corp agendas, paradox twists, or encounter scenes.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    with st.form("chat_generate_form"):
        chat_prompt = st.text_area(
            "Prompt",
            height=85,
            placeholder="Generate a Session 1 run involving a rogue technomancer and a Tradition cabal.",
            disabled=st.session_state.lock_chat,
        )
        generate_chat = st.form_submit_button(
            "⚡ Generate reply",
            disabled=st.session_state.lock_chat,
            use_container_width=True,
        )

    if generate_chat and chat_prompt.strip():
        if not api_key:
            st.warning("Add your API key in the sidebar before chatting.", icon="🗝️")
            st.stop()

        user_prompt = chat_prompt.strip()
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        log_event("User generated chat request")

        request_messages = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
        ]

        try:
            with transcript:
                with st.chat_message("user"):
                    st.markdown(user_prompt)
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
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            log_event(f"Request failed: {exc}")
            st.error(f"Request failed: {exc}")


# -------------------------------
# Foundry Project
# -------------------------------
with foundry_tab:
    st.subheader("Foundry Project Forge")
    st.caption("Foundry campaign scaffolding with SR5e and M:tA as tone/system references, not forced crossover.")
    st.session_state.lock_foundry = st.toggle("🔒 Lock project settings", value=st.session_state.lock_foundry)

    a, b, c = st.columns(3)
    project_name = a.text_input("Project name", "Arcology of Broken Spheres", disabled=st.session_state.lock_foundry)
    game_system = b.selectbox(
        "Primary rules base",
        ["Shadowrun 5e", "Mage: The Ascension", "Other"],
        index=0,
        disabled=st.session_state.lock_foundry,
    )
    secondary_inspiration = c.selectbox(
        "Secondary inspiration",
        ["None", "Shadowrun 5e", "Mage: The Ascension", "Cyberpunk genre", "Urban occult genre"],
        index=2,
        disabled=st.session_state.lock_foundry,
    )

    d, e, f = st.columns(3)
    play_style = d.selectbox(
        "Play blend",
        ["Investigation-heavy", "Balanced", "Tactical ops", "Political occult thriller"],
        index=0,
        disabled=st.session_state.lock_foundry,
    )
    campaign_scope = e.selectbox(
        "Campaign scope",
        ["One-shot", "Mini-campaign", "Long campaign", "Open table"],
        index=2,
        disabled=st.session_state.lock_foundry,
    )
    session_length = f.select_slider("Session length", options=[2, 3, 4, 5, 6], value=4, disabled=st.session_state.lock_foundry)

    g1, g2 = st.columns(2)
    scene_style = g1.selectbox(
        "Scene style",
        ["Rain-soaked noir", "Neon action", "Occult surrealism", "Corporate cold realism"],
        index=0,
        disabled=st.session_state.lock_foundry,
    )

    modules_focus = g2.multiselect(
        "Foundry modules focus",
        ["Scene transitions", "Journal automation", "Dice overlays", "Audio ambience", "Clock trackers", "Token vision tools"],
        default=["Journal automation", "Clock trackers", "Audio ambience"],
        disabled=st.session_state.lock_foundry,
    )

    h1, h2, h3 = st.columns(3)
    noir = h1.slider("Noir dial", 0, 10, 8, disabled=st.session_state.lock_foundry)
    horror = h2.slider("Horror dial", 0, 10, 6, disabled=st.session_state.lock_foundry)
    hope = h3.slider("Hope dial", 0, 10, 4, disabled=st.session_state.lock_foundry)

    i1, i2, i3 = st.columns(3)
    city_layer = i1.selectbox(
        "City layer",
        ["Street-level survival", "Mid-tier syndicate game", "AAA corp warfare", "Global metaplot breach"],
        index=1,
        disabled=st.session_state.lock_foundry,
    )
    corp_heat = i2.selectbox(
        "Corporate heat",
        ["Cold", "Simmering", "Hostile", "Black-ops live"],
        index=2,
        disabled=st.session_state.lock_foundry,
    )
    paradigm_pressure = i3.selectbox(
        "Paradox pressure",
        ["Subtle", "Noticeable", "Dangerous", "Reality-breaking"],
        index=1,
        disabled=st.session_state.lock_foundry,
    )

    j1, j2, j3 = st.columns(3)
    metaplot_thread = j1.selectbox(
        "Metaplot thread",
        ["Sleeper unrest", "Technocratic consolidation", "Dragon-backed covert war", "Nephandi infiltration"],
        disabled=st.session_state.lock_foundry,
    )
    legwork_density = j2.slider("Legwork density", 1, 10, 8, disabled=st.session_state.lock_foundry)
    combat_density = j3.slider("Combat density", 1, 10, 5, disabled=st.session_state.lock_foundry)

    k1, k2 = st.columns(2)
    safety_tools = k1.multiselect(
        "Safety tools",
        ["Lines & Veils", "X-Card", "Open Door", "Script Change", "Stars & Wishes"],
        default=["Lines & Veils", "Open Door", "Stars & Wishes"],
        disabled=st.session_state.lock_foundry,
    )
    session_zero_focus = k2.selectbox(
        "Session zero focus",
        ["Runner team trust", "Tradition cabal dynamics", "Cross-faction debts", "Boundaries and themes"],
        disabled=st.session_state.lock_foundry,
    )

    l1, l2 = st.columns(2)
    opening_district = l1.text_input("Opening district", "Glowmarket Barrens", disabled=st.session_state.lock_foundry)
    signature_antagonist = l2.text_input("Signature antagonist", "Director Vale of Parallax Biodyne", disabled=st.session_state.lock_foundry)

    premise = st.text_area(
        "Core premise",
        value="A megacorp's resonance-harvesting grid collides with a collapsing chantry node, and both the matrix and consensus reality start bleeding into each other.",
        height=88,
        disabled=st.session_state.lock_foundry,
    )

    fog_of_war = st.toggle("Enable Fog of War", value=True, disabled=st.session_state.lock_foundry)
    dynamic_lighting = st.toggle("Enable Dynamic Lighting", value=True, disabled=st.session_state.lock_foundry)

    foundry_data = {
        "project_name": project_name,
        "game_system": game_system,
        "play_style": play_style,
        "secondary_inspiration": secondary_inspiration,
        "campaign_scope": campaign_scope,
        "session_length": session_length,
        "noir": noir,
        "horror": horror,
        "hope": hope,
        "city_layer": city_layer,
        "corp_heat": corp_heat,
        "paradigm_pressure": paradigm_pressure,
        "metaplot_thread": metaplot_thread,
        "modules_focus": modules_focus or ["Journal automation"],
        "scene_style": scene_style,
        "safety_tools": safety_tools or ["Open Door"],
        "session_zero_focus": session_zero_focus,
        "opening_district": opening_district,
        "signature_antagonist": signature_antagonist,
        "hook_bias": "Conspiracy legwork and occult fallout",
        "legwork_density": legwork_density,
        "combat_density": combat_density,
        "premise": premise,
        "fog_of_war": fog_of_war,
        "dynamic_lighting": dynamic_lighting,
    }

    c1, c2, c3 = st.columns(3)
    generate_blueprint = c1.button("⚡ Generate blueprint", use_container_width=True, disabled=st.session_state.lock_foundry)
    save_blueprint = c2.button("📥 Save blueprint", use_container_width=True, disabled=st.session_state.lock_foundry)
    bible_blueprint = c3.button("📚 Add to World Bible", use_container_width=True, disabled=st.session_state.lock_foundry)

    if generate_blueprint:
        st.session_state.foundry_blueprint = compose_blueprint(foundry_data)
        log_event(f"Blueprint generated: {project_name}")

    blueprint_preview = st.session_state.foundry_blueprint or compose_blueprint(foundry_data)
    st.markdown("#### Blueprint Preview")
    st.markdown(blueprint_preview)

    if save_blueprint:
        st.session_state.project_blueprints.append(
            {
                "name": project_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "blueprint": blueprint_preview,
            }
        )
        log_event(f"Project blueprint saved: {project_name}")
        st.success("Blueprint saved to project vault.")

    if bible_blueprint:
        st.session_state.world_bible.append(
            {
                "title": f"Campaign Blueprint: {project_name}",
                "tags": ["blueprint", "foundry", "shadowrun", "mage"],
                "content": blueprint_preview,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        log_event(f"Blueprint pushed to world bible: {project_name}")
        st.success("Blueprint added to World Bible.")


# -------------------------------
# World Bible
# -------------------------------
with world_tab:
    st.subheader("World Bible")
    st.session_state.lock_world = st.toggle("🔒 Lock world bible", value=st.session_state.lock_world)

    left, right = st.columns([1, 2])
    with left:
        entry_title = st.text_input("Entry title", disabled=st.session_state.lock_world)
        entry_tags = st.text_input("Tags (comma-separated)", disabled=st.session_state.lock_world)
        entry_seed = st.text_area(
            "Entry seed",
            height=80,
            placeholder="Example: A syndicate fixer who brokers favors between mages and runners.",
            disabled=st.session_state.lock_world,
        )
        generate_world = st.button("⚡ Generate lore draft", disabled=st.session_state.lock_world, use_container_width=True)

        if generate_world:
            draft_tags = [tag.strip() for tag in entry_tags.split(",") if tag.strip()]
            tag_line = ", ".join(draft_tags) if draft_tags else "shadowrun, mage, foundry"
            st.session_state.world_entry_draft = dedent(
                f"""
                **Canon Focus:** {entry_title or 'Untitled entry'}

                **Tags:** {tag_line}

                {entry_seed or 'No seed provided.'}

                **Conflict Hook:** A quiet alliance shatters when matrix anomalies reveal astral fingerprints.
                **GM Use:** Use as a recurring thread across 2-3 sessions.
                """
            ).strip()
            log_event(f"World draft generated: {entry_title or 'Untitled entry'}")

        entry_content = st.text_area(
            "Lore content",
            value=st.session_state.world_entry_draft,
            height=180,
            disabled=st.session_state.lock_world,
        )
        add_entry = st.button("Add entry", disabled=st.session_state.lock_world, use_container_width=True)

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
        entries_container = st.container(height=510, border=True)
        with entries_container:
            if not st.session_state.world_bible:
                st.info("No lore entries yet. Generate a draft or add your first canon note.")
            for idx, entry in enumerate(reversed(st.session_state.world_bible)):
                with st.expander(f"{len(st.session_state.world_bible)-idx}. {entry['title']}"):
                    st.write(f"**Tags:** {', '.join(entry['tags']) if entry['tags'] else 'None'}")
                    st.write(entry["content"])


# -------------------------------
# Prompt Forge
# -------------------------------
with prompt_tab:
    st.subheader("Prompt Forge")
    st.session_state.lock_prompt = st.toggle("🔒 Lock prompt forge", value=st.session_state.lock_prompt)

    col_a, col_b, col_c = st.columns(3)
    subject = col_a.text_input("Subject", "Hermetic sanctum hidden inside a flooded data center", disabled=st.session_state.lock_prompt)
    style = col_b.selectbox(
        "Style",
        ["Neo-noir concept art", "Occult cyberpunk matte painting", "Gritty comic panel", "Foundry scene illustration"],
        disabled=st.session_state.lock_prompt,
    )
    lighting = col_c.selectbox(
        "Lighting",
        ["Holographic haze", "Cold sodium streetlight", "Ritual candle + monitor glow", "Storm-lit rooftop"],
        disabled=st.session_state.lock_prompt,
    )

    col_d, col_e, col_f = st.columns(3)
    mood = col_d.selectbox("Mood", ["Paranoid", "Arcane tense", "Melancholic", "Predatory"], disabled=st.session_state.lock_prompt)
    camera = col_e.selectbox("Framing", ["Wide establishing", "Character close-up", "Top-down tactical", "Isometric"], disabled=st.session_state.lock_prompt)
    detail_dial = col_f.slider("Detail dial", 1, 10, 8, disabled=st.session_state.lock_prompt)

    negative = st.text_input("Negative prompt", "blurry, low detail, text artifacts, watermark", disabled=st.session_state.lock_prompt)

    generate_prompt = st.button("⚡ Generate prompt", disabled=st.session_state.lock_prompt, use_container_width=True)
    if generate_prompt:
        st.session_state.prompt_preview = (
            f"{subject}. Style: {style}. Lighting: {lighting}. Mood: {mood}. "
            f"Framing: {camera}. Detail level {detail_dial}/10. "
            f"Foundry-ready scene composition, layered depth cues. Negative prompt: {negative}."
        )
        log_event("Image prompt generated")

    preview = st.session_state.prompt_preview or (
        f"{subject}. Style: {style}. Lighting: {lighting}. Mood: {mood}. "
        f"Framing: {camera}. Detail level {detail_dial}/10. Negative prompt: {negative}."
    )
    st.code(preview, language="text")

    if st.button("Save prompt to vault", disabled=st.session_state.lock_prompt, use_container_width=True):
        st.session_state.prompt_vault.append(
            {
                "prompt": preview,
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


# -------------------------------
# Control Room
# -------------------------------
with admin_tab:
    st.subheader("Control Room")
    st.session_state.lock_admin = st.toggle("🔒 Lock control room", value=st.session_state.lock_admin)

    payload = {
        "messages": st.session_state.messages,
        "world_bible": st.session_state.world_bible,
        "prompt_vault": st.session_state.prompt_vault,
        "project_blueprints": st.session_state.project_blueprints,
        "audit_log": st.session_state.audit_log,
    }

    r1, r2 = st.columns(2)
    if r1.button("⚡ Generate workspace summary", use_container_width=True, disabled=st.session_state.lock_admin):
        st.session_state.admin_summary = dedent(
            f"""
            ### Workspace Summary
            - Messages: {len(st.session_state.messages)}
            - World Bible Entries: {len(st.session_state.world_bible)}
            - Prompt Vault Entries: {len(st.session_state.prompt_vault)}
            - Project Blueprints: {len(st.session_state.project_blueprints)}
            - Latest Event: {st.session_state.audit_log[-1] if st.session_state.audit_log else 'No activity yet'}
            """
        ).strip()
        log_event("Workspace summary generated")

    if st.session_state.admin_summary:
        st.markdown(st.session_state.admin_summary)

    r2.download_button(
        "Export workspace JSON",
        data=json.dumps(payload, indent=2),
        file_name="ttrpg_studio_export.json",
        mime="application/json",
        use_container_width=True,
        disabled=st.session_state.lock_admin,
    )

    uploaded = st.file_uploader("Import workspace JSON", type=["json"], disabled=st.session_state.lock_admin)
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
