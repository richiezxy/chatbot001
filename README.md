# 💬 Multi-provider chatbot template

A Streamlit chatbot with a cleaner interface and support for both **OpenAI** and **OpenRouter**.

## Features

- Provider switcher (OpenAI / OpenRouter)
- Configurable base URL and model
- System prompt + temperature controls
- Streaming assistant responses
- Persistent chat history in session state
- One-click clear conversation button

### How to run it on your own machine

1. Install the requirements

   ```bash
   pip install -r requirements.txt
   ```

2. Run the app

   ```bash
   streamlit run streamlit_app.py
   ```

3. In the sidebar:
   - Choose your provider.
   - Add the matching API key.
   - Optionally tune model, system prompt, and temperature.
