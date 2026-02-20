# 🧙 TTRPG Studio (OpenAI + OpenRouter)

A Streamlit project moving toward a **t3.chat-style** daily workspace for tabletop worldbuilding, image prompt design, and campaign operations.

## What this build now includes

- Multi-provider chat: **OpenAI** and **OpenRouter**
- TTRPG-tuned assistant modes (world builder, narrative designer, encounter architect)
- World Bible note management for lore continuity
- Image Prompt Forge + prompt vault storage
- Admin controls for JSON export/import and audit logging
- Optional visual theme: **"TechnoDruid 20%"**

## Run locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start app:

   ```bash
   streamlit run streamlit_app.py
   ```

3. In sidebar:
   - choose provider
   - enter API key
   - tune model + temperature + TTRPG mode

## Next milestones

See `ROADMAP.md` for phased evolution into a robust daily-use platform.


## Compatibility note

This app supports both modern and legacy OpenAI Python SDK styles at runtime.
If your environment has an older `openai` package (that does not expose `OpenAI`),
the app automatically falls back to the legacy `openai.ChatCompletion` flow.

Recommended upgrade for best long-term support:

```bash
pip install --upgrade openai
```
