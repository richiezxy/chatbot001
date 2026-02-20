import streamlit as st
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

        with st.chat_message("assistant"):
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as exc:  # noqa: BLE001
        st.error(f"Request failed: {exc}")
