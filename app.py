import os
import streamlit as st
from dotenv import load_dotenv

from rag import RAGIndex
from github_llm import chat_completion, GitHubModelsError

load_dotenv()

st.set_page_config(page_title="Chatbot que demuestra las capacidades de RAG y GitHub Models", layout="wide")

st.title("📚 Chatbot que usa RAG y GitHub Models (Preguntas y Respuestas)")
st.caption("Las respuestas se generan ÚNICAMENTE a partir de los documentos en /docs. Primero construye el índice.")

with st.sidebar:
    st.header("Configuración")
    top_k = st.slider("Número de fragmentos a recuperar (Top-k)", 2, 8, 4)
    max_tokens = st.slider("Máximo de tokens (longitud de la respuesta)", 200, 1200, 600, step=50)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.2, step=0.1)

    st.divider()
    st.subheader("Índice")
    if st.button("Construir / Reconstruir Índice"):
        try:
            rag = RAGIndex()
            rag.build(docs_dir="docs")
            rag.save(data_dir="data")
            st.success("Índice construido y guardado ✅")
        except Exception as e:
            st.error(str(e))

    if st.button("Cargar Índice Existente"):
        try:
            rag = RAGIndex()
            rag.load(data_dir="data")
            st.session_state["rag_ready"] = True
            st.success("Índice cargado ✅")
        except Exception as e:
            st.error(str(e))

# Asegurarse de que RAG esté disponible
if "rag" not in st.session_state:
    st.session_state["rag"] = RAGIndex()

rag: RAGIndex = st.session_state["rag"]

# Intentar auto-carga si existe data
if rag.index is None:
    try:
        rag.load("data")
    except Exception:
        pass

question = st.text_input(
    "Haz una pregunta sobre los materiales del curso:",
    placeholder="Ej., ¿Qué es el aprendizaje supervisado?"
)

col1, col2 = st.columns([1, 1])
with col1:
    ask = st.button("Preguntar")
with col2:
    show_sources = st.checkbox("Mostrar fuentes", value=True)

if ask and question.strip():
    try:
        retrieved = rag.retrieve(question, top_k=top_k)
        if not retrieved:
            st.warning("No encontré información relevante en los documentos.")
        else:
            context_blocks = []
            for ch, score in retrieved:
                context_blocks.append(
                    f"[FUENTE: {ch.source} | fragmento {ch.chunk_id} | puntuación {score:.3f}]\n{ch.text}"
                )
            context = "\n\n---\n\n".join(context_blocks)

            system = (
                "Eres un asistente del curso. Responde SOLO usando el CONTEXTO proporcionado en los documentos.\n"
                "Reglas:\n"
                "1) Si la respuesta no está en el CONTEXTO, di: \"No pude encontrar esa información en los documentos del curso.\" \n"
                "2) No uses conocimiento externo.\n"
                "3) Cita las fuentes referenciando [FUENTE: archivo | fragmento] en tu respuesta.\n"
                "4) Mantén la respuesta clara y concisa.\n"
            )

            user = f"PREGUNTA:\n{question}\n\nCONTEXTO:\n{context}"

            answer = chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            st.subheader("Respuesta")
            st.write(answer)

            if show_sources:
                st.subheader("Fuentes Recuperadas")
                for ch, score in retrieved:
                    with st.expander(f"{ch.source} | fragmento {ch.chunk_id} | puntuación {score:.3f}"):
                        st.write(ch.text)

    except GitHubModelsError as e:
        st.error(f"Error de la API del modelo: {e}")
    except Exception as e:
        st.error(str(e))

st.divider()
st.markdown(
    """
### Cómo usar
1. Coloca los documentos del curso en la carpeta **docs/** (PDF, DOCX, TXT).
2. Haz clic en **Construir / Reconstruir Índice** (barra lateral).
3. Haz preguntas y verifica que las respuestas incluyan las fuentes.

**Tip:** Haz una pregunta que NO esté en los documentos para ver cómo el chatbot rechaza correctamente la respuesta.
"""
)
