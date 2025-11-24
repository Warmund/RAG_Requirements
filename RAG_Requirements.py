import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import load_index_from_storage
from llama_index.llms.lmstudio import LMStudio
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

# --- Streamlit Konfiguration ---
st.set_page_config(page_title="Q&A Assistent")
st.title("RAG Versuch")

# Session-State initialisieren
if "step" not in st.session_state:
    st.session_state.step = "fragen"  # Startschritt
if "antworten" not in st.session_state:
    st.session_state.antworten = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "options" not in st.session_state:
    st.session_state.options = []

chat_memory = ChatMemoryBuffer.from_defaults(token_limit=10000)

def build_system_prompt():
    base_prompt = (
        "Du arbeitest als Projektmanager und musst ein Pflichtenheft erstellen. "
        "Nutze dafür die Vorlage, welche dir gegeben ist. "
        "Verwende auch die Informationen aus alten Pflichtenheften, um das neue Pflichtenheft zu erstellen.\n\n"
    )
    if st.session_state.antworten:
        antworten_text = "\n".join(
            f"- {frage}: {antwort}" for frage, antwort in st.session_state.antworten.items()
        )
        base_prompt += f"Hier sind die Antworten auf vorherige Fragen:\n{antworten_text}\n"

    if st.session_state.options:
        base_prompt += f"Du sollst in dieser Antwort folgendes bearbeiten: {', '.join(map(str, st.session_state.options))}. Schreibe einen längeren Abschnitt darüber, nach deiner Vorlage"

    return base_prompt


# --- Modelle initialisieren ---
@st.cache_resource
def initialize_models():
    llm = LMStudio(
        model_name="openai/gpt-oss-20b",
        base_url="http://localhost:1234/v1",
        temperature=0.7,
        request_timeout=300,
        system_prompt=build_system_prompt()
    )
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    return llm, embed_model

llm, embed_model = initialize_models()

# --- Index laden oder erstellen ---
@st.cache_resource
def load_index():
    persist_path = "./storage"
    if os.path.exists(persist_path):
        storage_context = StorageContext.from_defaults(persist_dir=persist_path)
        index = load_index_from_storage(storage_context)
    else:
        with st.spinner("Index wird erstellt..."):
            documents = SimpleDirectoryReader("./docs").load_data()
            index = VectorStoreIndex.from_documents(documents)
            os.makedirs(persist_path, exist_ok=True)
            index.storage_context.persist()
            st.success("Index erstellt und gespeichert!")
    return index

index = load_index()

@st.cache_resource
def get_chat_engine(_index):
    return _index.as_chat_engine(
        chat_mode="context",
        memory=chat_memory,
        llm=llm,
        streaming=False,
    )
chat_engine = get_chat_engine(index)


# --- Schritt 1: Fragen beantworten ---
fragen = [
    "Beschreibe kurz das Konzept des Projekts.",
    "Welche Ziele sollen mit dem Projekt erreicht werden?",
    "Welche Zielgruppe soll mit dem Projekt angesprochen werden?",
    "Kannst Du Anforderungen nennen welche auf jeden Fall gegeben sein müssen?",
    "Kannst Du die Rahmenbedingungen erläutern? Stichpunkte wie Budget, Zeitrahmen, Ressourcen",
    "Kannst Du etwas über die Liefer- und Abnahmebedingungen sagen?",
    "Generelle Informationen welche ich wissen sollte: "
]

if st.session_state.step == "fragen":
    st.header("Bitte beantworten Sie zuerst die Fragen")
    for frage in fragen:
        if frage not in st.session_state.antworten:
            st.session_state.antworten[frage] = ""
        st.session_state.antworten[frage] = st.text_input(frage, value=st.session_state.antworten[frage])

    if st.button("Weiter zum Chat"):
        # Prüfen, ob alle Fragen beantwortet sind
        if all(st.session_state.antworten[f] != "" for f in fragen):
            llm.system_prompt = build_system_prompt()
            st.session_state.step = "chat"
            st.session_state["rerun_flag"] = True
            st.rerun()

        else:
            st.warning("Bitte alle Fragen beantworten, bevor Sie fortfahren.")

# --- Schritt 2: Chat Interface ---
elif st.session_state.step == "chat":
    st.header("Chat mit LLM")
    st.markdown("Wähle was erstellt werden soll: (Optional)")
    option_labels = ["Komplettes Pflichtenheft", "Allgemeines", "Konzept", "Funktionale Anforderungen", "Nicht-funktionale Anforderungen", "Rahmenbedingungen", "Liefer- und Abnahmebedingungen"]
    selected_options = []

    cols = st.columns(len(option_labels))
    for idx, (col, label) in enumerate(zip(cols, option_labels)):
        if col.button(label):
            st.session_state.options = []
            st.session_state.options.append(label)
            st.session_state.messages.append({"role": "system", "content": f"{label} wurde gewählt."})
            llm.system_prompt = build_system_prompt()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Schreibe \"Go\" um zu starten oder spezifiziere genauer was Du möchtest...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Antwort wird generiert..."):
                try:
                    response = chat_engine.chat(user_question)
                    response_text = response.response
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.session_state.options = []
                    llm.system_prompt = build_system_prompt()
                except Exception as e:
                    error_message = f"Fehler: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

    with st.sidebar:
        st.header("Zusätzliche Optionen")
        uploaded_file = st.file_uploader("Upload eines bestimmten Lastenhefts, welches besonders hilfreich sein kann. (Optional)", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            st.success(f"Datei {uploaded_file.name} erfolgreich hochgeladen!")
            file_content = uploaded_file.read().decode("utf-8")
            # Dateiinhalt in den System-Prompt einfügen
            llm.system_prompt += f"\nHier ist ein vorheriges Latenheft, welches der Nutzer als besonders Hilfreich für dich sieht, beziehe dieses in deine Antwort mit ein:\n{file_content}\n"