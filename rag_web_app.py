import streamlit as st
from src.rag_system import SimpleRAG
from config.config import Config

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.markdown("<h2 style='text-align: center; color: #2c3e50;'>💬 Policy Chatbot</h2>", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    rag = SimpleRAG()
    rag.setup_rag_system(Config.DOCUMENTS_PATH, VECTOR_STORE_PATH=None)
    return rag

rag = load_rag_system()

st.markdown("""
<div style='text-align: center; font-size: 18px; color: #555;'>
Ask about policies: remote work, employee break, sick days and sick leave, employee referral bonus program, loyality rewards vacation days, 
physical security, server security. building security, web application security, workstation security, remote access, password protection, 
password construction guidelines, email and ethics.  .<br>
</div>
""", unsafe_allow_html=True)

st.markdown("---")  

st.markdown("""
<style>
    /* Background */
    .stApp {
        background-color: #f4f6f9;
    }

    /* Chat message styling */
    .chat-bubble {
        padding: 12px 20px;
        margin: 10px 0;
        border-radius: 12px;
        max-width: 80%;
    }
    .user-msg {
        background-color: #d1ecf1;
        align-self: flex-end;
        color: #0c5460;
    }
    .assistant-msg {
        background-color: #e2e3e5;
        color: #383d41;
    }

    /* Center title */
    h1 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_input = st.chat_input("Ask question")

if user_input:
    #st.chat_message("user", avatar="👤").write(user_input)

    #with st.spinner("Analyzing..."):
       # response = rag.ask_question(user_input)

    with st.spinner("Generating answer..."):
        response = rag.ask_question(user_input)


    st.session_state.chat_history.append({
        "question": user_input,
        "answer": response["answer"],
        "sources": response["sources"]
    })

for msg in st.session_state.chat_history:
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"""
        <div class="chat-bubble user-msg">{msg["question"]}</div>
        """, unsafe_allow_html=True)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"""
        <div class="chat-bubble assistant-msg">{msg["answer"]}</div>
        """, unsafe_allow_html=True)

        if msg["sources"]:
            st.markdown("#### 📚 Sources:")
            from collections import Counter
            counts = Counter(msg["sources"])
            for src, count in counts.items():
                st.write(f"- {src} (x{count})")

with st.sidebar:
    st.markdown("## 🕘 Chat History")

    for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # zadnjih 10
        with st.expander(f"💬 {chat['question'][:40]}..."):
            st.write(f"**You:** {chat['question']}")
            st.write(f"**Chatbot:** {chat['answer']}")
            if chat["sources"]:
                st.write("**Sources:**")
                for src in chat["sources"]:
                    st.write(f"- {src}")


if st.button("🗑️ Reset Chat"):
    st.session_state.chat_history = []
    st.rerun()





#question = st.text_input("🧠 Question")

#if st.button("Answer") and question:
 #   result = rag.ask_question(question)

  #  st.markdown("### 🧾 Answer")
   # st.write(result["answer"])

    #st.markdown("### 📚 Sources")
    #for src in set(result["sources"]):

     #   st.write(f"- {src}")"""
