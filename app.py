import json
import os
from datetime import datetime

import requests
import streamlit as st

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

st.set_page_config(page_title="My AI Chat", layout="wide")

st.title("My AI Chat")
st.caption(
    "This app uses the Hugging Face Inference Router with an OpenAI-compatible chat completions API. "
    "The free tier is rate-limited, so responses may be slow or occasionally fail."
)

# Load token from Streamlit secrets. Do not hardcode it.
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    hf_token = None

if not hf_token or not str(hf_token).strip():
    st.error(
        "Missing Hugging Face token. Add `HF_TOKEN` to `.streamlit/secrets.toml` "
        "and restart the app."
    )
    st.stop()

# Initialize chat history and sidebar state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if "chats" not in st.session_state:
    st.session_state.chats = []
if "current_title" not in st.session_state:
    st.session_state.current_title = "New Chat"
if "memory" not in st.session_state:
    st.session_state.memory = {
        "hiking": "true",
        "name": "User",
        "preferred_language": "English",
        "interests": ["outdoors", "history", "food"],
        "communication_style": "informative",
        "favorite_topics": ["California history", "food", "plants"],
    }


def load_memory_from_file() -> None:
    if os.path.exists("memory.json"):
        try:
            with open("memory.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                st.session_state.memory = data
        except (OSError, json.JSONDecodeError):
            pass


def save_memory_to_file() -> None:
    try:
        with open("memory.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.memory, f, indent=2)
    except OSError:
        pass


def start_new_chat() -> None:
    if len(st.session_state.messages) > 1:
        st.session_state.chats.append(
            {
                "title": st.session_state.current_title,
                "messages": st.session_state.messages,
                "timestamp": datetime.now().strftime("%b %d"),
            }
        )
    st.session_state.current_title = "New Chat"
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]


load_memory_from_file()

with st.sidebar:
    st.header("Chats")
    if st.button("New Chat", use_container_width=True):
        start_new_chat()

    with st.expander("User Memory", expanded=True):
        if st.button("Clear Memory", use_container_width=True):
            st.session_state.memory = {}
            save_memory_to_file()
        st.json(st.session_state.memory)

    st.divider()
    st.caption("Recent Chats")
    if not st.session_state.chats:
        st.caption("No recent chats yet.")
    else:
        for i, chat in enumerate(reversed(st.session_state.chats), start=1):
            label = f"{chat['title']} — {chat['timestamp']}"
            if st.button(label, key=f"chat_{i}", use_container_width=True):
                st.session_state.current_title = chat["title"]
                st.session_state.messages = chat["messages"]

# Render chat history (skip system in UI)
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Fixed input bar at the bottom
user_input = st.chat_input("Type a message and press Enter")

if user_input:
    user_message = {"role": "user", "content": user_input.strip()}
    if st.session_state.current_title == "New Chat":
        st.session_state.current_title = user_message["content"][:32] or "New Chat"
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.write(user_message["content"])

    payload = {
        "model": HF_MODEL,
        "messages": st.session_state.messages,
        "temperature": 0.7,
        "max_tokens": 256,
    }
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    with st.spinner("Contacting Hugging Face Inference Router..."):
        try:
            resp = requests.post(HF_ENDPOINT, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                assistant_text = data["choices"][0]["message"]["content"]
                assistant_message = {"role": "assistant", "content": assistant_text}
                st.session_state.messages.append(assistant_message)
                with st.chat_message("assistant"):
                    st.write(assistant_text)
            elif resp.status_code == 401:
                st.error("Unauthorized. Check that your HF token is valid and has access.")
            elif resp.status_code == 429:
                st.warning("Rate limit reached. Please wait a bit and try again.")
            else:
                st.error(f"Request failed with status {resp.status_code}.")
                st.code(resp.text)
        except requests.exceptions.Timeout:
            st.warning("The request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            st.error("Network error while contacting the API.")
            st.code(str(e))
