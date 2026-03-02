# Agentic‑Healthcare‑Assistant

**Agentic‑Healthcare‑Assistant** is an autonomous, intelligent healthcare AI assistant built with **agentic AI architecture**. It enables **intelligent symptom analysis**, **multilingual voice/text interaction**, and **preliminary medical guidance** for users in both **urban and rural settings**. The system leverages **Python, LangChain, RAG, FAISS, machine learning (ML), and natural language processing (NLP)** to provide a more grounded, reliable, and interactive healthcare support experience. 

---

## 🧠 Project Overview

The core innovation in this project is **`mediagent`** — the agentic AI chatbot that drives the entire healthcare assistant. What makes it novel is:

### 🔥 Novelty (MediAgent)

* **Agentic AI Chatbot Core** — It orchestrates a collection of agent workflows using LangChain and intelligent decision logic.
* **Retrieval‑Augmented Generation (RAG)** — Instead of just prompting a large language model (LLM), this agent uses a vector store (FAISS) and retrieval techniques to ground responses in evidence and reduce hallucinations.
* **Dynamic Mode Selection** — Based on user intent, the agent intelligently decides whether to retrieve relevant medical context from the knowledge base or generate conversational responses.
* **Multi‑Step Reasoning** — By splitting tasks (e.g., symptom analysis → disease inference → recommendation), it mirrors human‑like reasoning rather than single‑shot LLM completion.
* **Multimodal Interaction Support** — It supports voice/text interaction by integrating audio handling logic (for voice input/output), enabling more inclusive user access.
  (This kind of architecture is inspired by agentic workflows seen in other healthcare assistants — multi‑agent orchestration and RAG pipelines commonly improve accuracy and applicability in healthcare AI systems.) ([GitHub][2])

---
<img width="843" height="648" alt="image" src="https://github.com/user-attachments/assets/7c4190e9-72fe-4254-9822-e79231646121" />

## 🧩 Key Features

### 🧪 Intelligent Symptom Analysis

* Users can describe symptoms naturally in text or voice.
* The system interprets, embeds, and routes queries to the proper reasoning path.

### 🌐 Multilingual Voice/Text Interaction

* Supports interactions in multiple languages (e.g., English plus regional languages).
* Converts speech to text and back using audio processing modules.

### 📚 Retrieval‑Augmented Generation (RAG)

* Uses FAISS to store semantic embeddings of medical knowledge.
* Retrieves relevant medical data before generating an answer — grounding responses.

### 🧠 MediAgent Central Logic

* Central entry point for agent execution, decision routing, and response generation.
* Handles fallback logic: retrieval first → generation → clarification questions.

## 📁 Project Structure

```
.
├── Disease Prediction and recommendation
│   ├── disease_prediction_system.ipynb
│   └── readme.md
├── Dockerfile
├── Drug Recommendation
│   ├── Drug_Recommendation.ipynb
│   └── readme.md
├── constraints.txt
├── data
│   ├── Disease-Prediction-and-Medical dataset
│   │   ├── Symptom-severity.csv
│   │   ├── Training.csv
│   │   ├── description.csv
│   │   ├── diets.csv
│   │   ├── medications.csv
│   │   ├── precautions_df.csv
│   │   ├── symptoms_df.csv
│   │   └── workout_df.csv
│   └── Drug reccomendation
│       └── medicine.csv
├── heart_disease_risk_assessment
│   ├── Data_Wrangling_pre_processing_notebook.ipynb
│   ├── Exploratory_data_analysis.ipynb
│   ├── Modeling.ipynb
│   └── readme.md
├── home.py
├── medibot_logs.jsonl
├── medibot_reply.mp3
├── medibot_voice.mp3
├── models
│   ├── first_feature_models
│   │   └── RandomForest.pkl
│   ├── second_feature_models
│   │   ├── medicine_dict.pkl
│   │   └── similarity.joblib
│   └── third_feature_models
│       ├── best_model.pkl
│       ├── best_model1.pkl
│       ├── brfss2022_data_wrangling_output.zip
│       ├── cbe_encoder.pkl
│       └── cbe_encoder1.pkl
├── pages
│   ├── 1_Disease-Prediction-and-medical-recommendation.py
│   ├── 2_drug_recommendation.py
│   ├── 3_heart_Disease_Risk_Assesment.py
│   └── mediagent.py
├── requirements.txt
├── response.mp3
├── test_token.py
├── utils
│   ├── AI Powered Healthcare System.mp4
│   ├── heart_disease.jpg
│   ├── img1.png
│   ├── img2.png
│   ├── img3.png
│   ├── img4.png
│   ├── img5.png
│   ├── img6.png
│   ├── img7.png
│   ├── img8.png
│   ├── medss.png
│   ├── ph1.png
│   ├── ph2.png
│   ├── ph3.png
│   ├── ph4.png
│   ├── ph5.png
│   ├── style.css
│   └── style_v1.css
└── vectorstore
    └── db_faiss
        ├── index.faiss
        └── index.pkl
```

## 🛠️ Tech Stack

| Component                  | Purpose                                      |               |
| -------------------------- | -------------------------------------------- | ------------- |
| **Python**                 | Primary programming language                 |               |
| **LangChain**              | Building agent workflows and LLM interaction |               |
| **FAISS**                  | Semantic vector search for RAG               |               |
| **OpenAI / LLM**           | Natural language reasoning and generation    |               |
| **NLP & ML Libraries**     | Embeddings, tests, risk assessment models    |               |
| **Voice Processing Tools** | Speech‑to‑Text & Text‑to‑Speech support      |               |

---

## 🧑‍💻 Getting Started

### ⚙️ Prerequisites

* Python 3.9+
* API keys for any LLM services you choose (e.g., OpenAI)
* Microphone/speaker for voice input/output (optional)

<h2>⚙️ Installation & Setup</h2>

<h3>1️⃣ Clone the Repository</h3>
<pre>
git clone https://github.com/AbhaySingh71/AI-Powered-Healthcare-Intelligence-System.git
cd AI-Powered-Healthcare-Intelligence-System
</pre>

<h3>2️⃣ Set Up the Virtual Environment</h3>
<pre>
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
</pre>

<h3>3️⃣ Install Dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<h3>4️⃣ Set Up Environment Variables</h3>
<p>Create a <code>.env</code> file and add:</p>
<pre>
**Set up environment variables**
   Create a `.env` file
  # .env
    * OPENAI_API_KEY=xxxx
    * GROQ_API_KEY=xxxxx
    * GOOGLE_MAPS_API_KEY =xxxx
   * Provide this keys in env file
</pre>
<p>Ensure it is added to GitHub Secrets when deploying.</p>

<h3>5️⃣ Run the Application</h3>
<pre>
streamlit run home.py
</pre>

---


## 🚀 Usage

### 💬 Run the Chatbot

Depending on setup (CLI or Web UI):

#### Command‑line

```bash
python home.py
```
#### For MediAgent

```bash
python mediagent.py
```
Once running, interact by text or voice prompts:

> “I’m feeling chest pain and dizziness — what might this be?”
> Agent routes to appropriate reasoning, logs context, retrieves supporting info, and returns a grounded guidance answer.

---
