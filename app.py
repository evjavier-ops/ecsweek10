import json
import os
import time
from datetime import datetime
from uuid import uuid4

import requests
import streamlit as st

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CHATS_DIR = "chats"

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

os.makedirs(CHATS_DIR, exist_ok=True)


def create_chat(title: str = "New Chat") -> dict:
    return {
        "id": str(uuid4()),
        "title": title,
        "messages": [],
        "timestamp": datetime.now().strftime("%b %d"),
    }


def chat_file_path(chat_id: str) -> str:
    return os.path.join(CHATS_DIR, f"{chat_id}.json")


def save_chat_to_file(chat: dict) -> None:
    try:
        with open(chat_file_path(chat["id"]), "w", encoding="utf-8") as f:
            json.dump(chat, f, indent=2)
    except OSError:
        pass


def delete_chat_file(chat_id: str) -> None:
    try:
        os.remove(chat_file_path(chat_id))
    except OSError:
        pass


def load_chats_from_disk() -> list[dict]:
    chats = []
    for name in os.listdir(CHATS_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(CHATS_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and {"id", "title", "messages"}.issubset(data.keys()):
                data["messages"] = [
                    m for m in data.get("messages", []) if m.get("role") != "system"
                ]
                chats.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    return chats


# Initialize chat state
if "chats" not in st.session_state:
    st.session_state.chats = load_chats_from_disk()
    if not st.session_state.chats:
        new_chat = create_chat()
        st.session_state.chats = [new_chat]
        save_chat_to_file(new_chat)
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = st.session_state.chats[0]["id"]

# Initialize memory state
if "memory" not in st.session_state:
    st.session_state.memory = {}


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


def merge_memory(new_data: dict) -> None:
    if not isinstance(new_data, dict):
        return
    for key, value in new_data.items():
        if value in (None, "", [], {}):
            continue
        st.session_state.memory[key] = value
    save_memory_to_file()


def extract_memory_from_message(user_text: str) -> None:
    if not user_text.strip():
        return
    prompt = (
        "Given this user message, extract any personal facts or preferences as a JSON object. "
        "If none, return {}. Only return valid JSON with no extra text."
    )
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(HF_ENDPOINT, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            try:
                extracted = json.loads(content)
                merge_memory(extracted)
            except json.JSONDecodeError:
                pass
    except requests.exceptions.RequestException:
        pass


load_memory_from_file()

# Sidebar UI
with st.sidebar:
    st.header("Chats")
    if st.button("New Chat", use_container_width=True):
        new_chat = create_chat()
        st.session_state.chats.append(new_chat)
        st.session_state.active_chat_id = new_chat["id"]
        save_chat_to_file(new_chat)

    with st.expander("User Memory", expanded=True):
        if st.button("Clear Memory", use_container_width=True):
            st.session_state.memory = {}
            save_memory_to_file()
        st.json(st.session_state.memory)

    st.divider()
    st.caption("Recent Chats")

    delete_id = None
    if not st.session_state.chats:
        st.caption("No chats yet.")
    else:
        for chat in st.session_state.chats:
            is_active = chat["id"] == st.session_state.active_chat_id
            label_prefix = "▶ " if is_active else ""
            label = f"{label_prefix}{chat['title']} — {chat['timestamp']}"
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                if st.button(label, key=f"select_{chat['id']}", use_container_width=True):
                    st.session_state.active_chat_id = chat["id"]
            with col2:
                if st.button("✕", key=f"delete_{chat['id']}"):
                    delete_id = chat["id"]

    if delete_id is not None:
        st.session_state.chats = [c for c in st.session_state.chats if c["id"] != delete_id]
        delete_chat_file(delete_id)
        if st.session_state.active_chat_id == delete_id:
            st.session_state.active_chat_id = (
                st.session_state.chats[0]["id"] if st.session_state.chats else None
            )
        st.rerun()


# Active chat lookup
active_chat = None
for chat in st.session_state.chats:
    if chat["id"] == st.session_state.active_chat_id:
        active_chat = chat
        break

if active_chat is None:
    st.info("No active chat. Create a new chat to begin.")
    st.stop()

# Render chat history
for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Fixed input bar at the bottom
user_input = st.chat_input("Type a message and press Enter")

if user_input:
    user_message = {"role": "user", "content": user_input.strip()}
    if active_chat["title"] == "New Chat":
        active_chat["title"] = user_message["content"][:32] or "New Chat"
    active_chat["messages"].append(user_message)
    save_chat_to_file(active_chat)
    with st.chat_message("user"):
        st.write(user_message["content"])

    memory_text = json.dumps(st.session_state.memory, ensure_ascii=True, indent=2)
    system_prompt = (
        "You are a helpful assistant. Use the user memory below to personalize replies. "
        "If memory is empty, respond normally.\n\nUser Memory:\n"
        f"{memory_text}"
    )

    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, *active_chat["messages"]],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    with st.spinner("Contacting Hugging Face Inference Router..."):
        try:
            resp = requests.post(
                HF_ENDPOINT, json=payload, headers=headers, timeout=60, stream=True
            )
            if resp.status_code == 200:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    full_text = ""
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line.replace("data:", "", 1).strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            chunk = delta.get("content", "")
                        except (KeyError, json.JSONDecodeError, IndexError, TypeError):
                            chunk = ""
                        if chunk:
                            full_text += chunk
                            placeholder.write(full_text)
                            time.sleep(0.02)

                assistant_message = {"role": "assistant", "content": full_text}
                active_chat["messages"].append(assistant_message)
                save_chat_to_file(active_chat)
                extract_memory_from_message(user_message["content"])
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
