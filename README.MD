# 📚 Taller de Procesamiento de Lenguaje Natural (RAG + GitHub Models)

Este proyecto es un **chatbot educativo** que responde preguntas **únicamente a partir de documentos proporcionados**.

Está diseñado para:
- Profesores e instructores
- Talleres de Procesamiento de Lenguaje Natural (NLP)
- Cursos universitarios
- Demostraciones prácticas de IA generativa

El chatbot utiliza:
- **Retrieval-Augmented Generation (RAG)**
- **GitHub Models** (uso gratuito para fines educativos)
- Una interfaz sencilla con **Streamlit**

---

## ✨ ¿Qué hace este chatbot?

✅ Responde preguntas solo usando los documentos cargados  
✅ Rechaza preguntas fuera del contenido del curso  
✅ Cita la fuente del documento utilizado en cada respuesta  
✅ Fácil de personalizar (solo cambiar documentos)  
✅ Se ejecuta localmente en el computador  

---

## 🧠 ¿Cómo funciona? (Nivel conceptual)

1. El profesor coloca los documentos del curso en la carpeta `docs/`
2. La aplicación genera representaciones vectoriales (embeddings)
3. Cuando un estudiante hace una pregunta:
   - Se buscan los fragmentos más relevantes
   - El modelo genera la respuesta usando **solo esos fragmentos**

Esto evita respuestas inventadas (alucinaciones) y mantiene el contenido controlado.

---

## 🛠 Requisitos

- Python **3.9 – 3.14**
- Cuenta de GitHub
- Conexión a Internet (para el modelo)

---

## 🚀 Instalación (Windows / macOS / Linux)

### 1️⃣ Descargar o clonar el repositorio

```bash
git clone https://github.com/rczalez/Taller-NLP
```
### 2️⃣ Crear y activar un entorno virtual
**Windows (PowerShell o CMD)**

```bash
python -m venv .venv
.venv\Scripts\activate
```
**macOS / Linux / Git Bash**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

## 🔐 Configuración de GitHub Models

### 1️⃣ Crear un token de GitHub

- GitHub → **Settings**
-  **Developer settings**
- **Personal access tokens** → *Fine-grained token*

- Activar permiso: **Models (Read)**

### 2️⃣ Crear archivo .env

Copia el archivo `env.example` y renómbralo como `.env`:

```bash
GITHUB_TOKEN=tu_token_de_github
GITHUB_MODELS_ENDPOINT=https://models.inference.ai.azure.com/chat/completions
MODEL_NAME=Meta-Llama-3.1-8B-Instruct
```

### ▶️ Ejecutar la aplicación

```bash
python -m streamlit run app.py
```
La aplicación se abrirá automáticamente en:

http://localhost:8501


## 📂 Agregar documentos del curso

1. Coloca los archivos en la carpeta docs/:
- PDF (`.pdf`)
- Word (`.docx`)
- Texto (`.txt`)

2. En la barra lateral de la aplicación:

- Haz clic en **Build / Rebuild Index**

3. Realiza preguntas relacionadas con el contenido

⚠️ Si cambias los documentos, debes reconstruir el índice nuevamente.

## 🧪 Ejemplos de preguntas

✔️ *¿De que trata el libro?*

✔️ *¿Cuales son los requerimientos tecnicos para el lector?*

❌ *¿Quién ganó el último mundial?* → El chatbot rechazará la pregunta.

## 🤖 Modelos compatibles (GitHub Models)

Los siguientes modelos han sido probados y funcionan correctamente con esta plantilla:

| Modelo | Uso recomendado |
|------|----------------|
| **Meta-Llama-3.1-8B-Instruct** | Mejor calidad general |
| **Phi-3-small-8k-instruct** | Respuestas más rápidas |
| **Phi-3-medium-128k-instruct** | Documentos largos |
| **Mistral-7B-Instruct** | Alternativa equilibrada |

### Cambiar el modelo

Para utilizar un modelo diferente, edita el archivo `.env` y modifica la variable `MODEL_NAME`:

```env
MODEL_NAME=Meta-Llama-3.1-8B-Instruct
```
## 📁 Estructura del Proyecto

La organización del proyecto es la siguiente:

```text
.
├── app.py               # Interfaz principal con Streamlit
├── rag.py               # Lógica de indexación y recuperación (RAG)
├── github_llm.py        # Conexión con la API de GitHub Models
├── utils.py             # Carga y procesamiento de documentos
├── docs/                # Documentos del curso (PDF, DOCX, TXT)
├── data/                # Índice vectorial generado automáticamente
├── requirements.txt     # Lista de dependencias del proyecto
├── .env.example         # Plantilla de variables de entorno
└── README.md            # Documentación del proyecto
```
**Descripción general**

- app.py: Define la interfaz web del chatbot.
- rag.py: Implementa el flujo RAG (búsqueda + generación).
- github_llm.py: Encapsula la comunicación con GitHub Models.
- utils.py: Maneja la lectura de documentos del curso.
- docs/: Carpeta donde el profesor coloca el material del curso.
- data/: Se crea automáticamente al construir el índice.
- requirements.txt: Permite instalar todas las dependencias con un solo comando.
- .env.example: Guía para configurar credenciales sin exponerlas.

## 🎓 Uso educativo

Esta plantilla es ideal para:

- Talleres de NLP
- Cursos universitarios
- Prácticas controladas de IA
- Demostraciones de RAG

Los estudiantes solo pueden acceder al contenido definido por el profesor.

⚠️ Consideraciones

- Se requiere conexión a Internet
- La primera indexación puede tardar algunos minutos
- Documentos grandes aumentan el tiempo de procesamiento

## 📜 Licencia

Este proyecto se proporciona con fines educativos y académicos.

Puede adaptarse libremente para enseñanza e investigación.

## 🙌 Agradecimientos

- GitHub Models
- Streamlit
- FAISS
- Sentence Transformers