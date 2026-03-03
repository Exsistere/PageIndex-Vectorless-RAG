import streamlit as st
import requests
import uuid
import json
BACKEND_URL = "http://127.0.0.1:8000"
SUBMIT_URL = f"{BACKEND_URL}/submit"
CHAT_URL = f"{BACKEND_URL}/chat"
st.set_page_config(page_title="LLM Chatbot", layout="centered")
st.title("🤖 LLM Chatbot")

# -----------------------------
# Session state initialization
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "lead_details" not in st.session_state:
    st.session_state.lead_details = {
        "name": None,
        "email": None,
        "date": None,
        "time": None,
        "product": None,
    }
if "lead_history" not in st.session_state:
    st.session_state.lead_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id=str(uuid.uuid4())

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

if "audio_consumed" not in st.session_state:
    st.session_state.audio_consumed = False

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

if "tts_cache" not in st.session_state:
    st.session_state.tts_cache = {}  # msg_index -> audio_bytes

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "tree" not in st.session_state:
    st.session_state.tree = None
if "node_map" not in st.session_state:
    st.session_state.node_map = None
# -----------------------------
# Display chat history
# -----------------------------
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# File uploader (per turn)
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF (optional)",
    type=["pdf"],
    key="pdf_uploader"
)

if uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    response = requests.post(SUBMIT_URL, files=files)
    if response.status_code == 200:
        data = response.json()
        st.session_state.doc_id = data["doc_id"]
        st.session_state.tree = data["tree"]
        st.session_state.node_map = data["node_map"]
        st.success("PDF submitted successfully! It will be processed in the background.")
# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Ask something about sales or booking...")
print(st.session_state.node_map, st.session_state.tree)
if user_input:
    if not st.session_state.node_map or not st.session_state.tree:
        st.markdown("PDF Not Uploaded")
        st.stop() 
    # ---- Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

     
    # ---- Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data_to_send = {
                "query": user_input,
                "node_map": st.session_state.node_map,
                "tree": st.session_state.tree
            }
            # if st.session_state.doc_id:
            #     data_to_send["doc_id"] = st.session_state.doc_id
            try:
                json.dumps(data_to_send)
                print("JSON serializable ✅")
            except Exception as e:
                print("Serialization error:", e)
            response = requests.post(
                CHAT_URL,
                json=data_to_send,
            )
            print("Status:", response.status_code)
            print("Error body:", response.text)

            if response.status_code == 200:
                print("response recieved from backend",response.status_code)
                st.session_state.audio_consumed = True
                st.session_state.audio_bytes = None
                st.session_state.audio_key += 1
                result = response.json()
                # st.session_state.graph_state = result.get("state", st.session_state.graph_state)
                assistant_reply = (
                    f"Response: {result['Response']}"
                )


            else:
                assistant_reply = "❌ Backend error."

        st.markdown(assistant_reply)

    # ---- Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
        )
    # print("rerun hit")
    st.rerun()
