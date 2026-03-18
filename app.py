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

st.subheader("Test Message")
st.write("User: Hello!")

payload = {
    "model": HF_MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    "temperature": 0.7,
    "max_tokens": 128,
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
            message = data["choices"][0]["message"]["content"]
            st.subheader("Model Response")
            st.write(message)
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
