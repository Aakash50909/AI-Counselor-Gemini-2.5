# 🌿 Serenity AI: A Hybrid-Architecture Mental Health Companion

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![AI](https://img.shields.io/badge/GenAI-Gemini%202.5%20Flash-green)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

**Serenity AI** is a mental health support prototype designed to bridge the gap between cold, robotic chatbots and real human empathy. 

I realized that while large language models (LLMs) are smart, they can be unpredictable—sometimes even dangerous—in a crisis. I built Serenity to solve this using a **"Hybrid Brain"**: a fast, logical machine learning model that acts as a safety guard, and a generative AI that acts as a compassionate listener.

---

## How It Works (The Logic)

Think of Serenity like a hospital visit:

1.  **The Triage Nurse (Logistic Regression):** Before the user even speaks to the doctor, a fast, reliable system checks the symptoms. In my app, a Logistic Regression model instantly classifies the text into 7 categories (like *Anxiety*, *Depression*, or *Suicidal*).
2.  **The Specialist (Gemini + RAG):** Once the system knows *what* the user is feeling, it gives the AI specific medical instructions (protocols) on how to help. For example, if the user is anxious, it doesn't just say "calm down"—it guides them through a 5-4-3-2-1 grounding exercise.
3.  **The Safety Valve:** If the Triage Nurse detects a crisis (Self-harm/Suicide), the AI is physically disconnected. A hard-coded emergency protocol takes over to ensure safety.

---

## Engineering Decisions: Why Logistic Regression?

During my research phase, I tested several models including Naive Bayes, Decision Trees, and XGBoost.

> **Honest Technical Note:**
> While **XGBoost** was technically the best-performing model on paper (highest accuracy), I chose **Logistic Regression** for the live application.
>
> **Why?** In real-life demos and protocols, speed is everything. Logistic Regression gives the fastest, "somewhat accurate" results with microsecond latency, ensuring the user feels heard instantly.
>
> *Note: This project is currently a prototype. It is yet to be fine-tuned heavily to produce the desired, clinically perfect, and less erroneous responses I aim for in the future.*

---

## Tech Stack

| Component | Technology | Why? |
| :--- | :--- | :--- |
| **Frontend** | **Streamlit** | For a clean, "Glassmorphism" UI that feels calming. |
| **The Brain** | **Gemini 2.5 Flash** | Powerful reasoning with a large context window. |
| **The Guard** | **Logistic Regression** | Fast, interpretable, and lightweight. |
| **The Voice** | **Deepgram Aura** | "Asteria" model for human-like breathing and pacing. |
| **Processing** | **NLTK & Scikit-Learn** | TF-IDF vectorization and Stemming for NLP. |

---

## Future Roadmap

I am just getting started. To turn Serenity from a prototype into a product, here is my plan:

- [ ] **Heavy Fine-Tuning:** Currently, the system uses "Few-Shot Prompting." I plan to fine-tune a dedicated model (like Llama 3) on real counseling datasets to drastically reduce errors and hallucination.
- [ ] **Full RAG Integration:** Connect the backend to a Pinecone Vector Database containing verified Psychology textbooks, allowing the AI to cite sources.
- [ ] **Long-Term Memory:** Implement session handling so Serenity remembers mood trends from yesterday, creating a continuous bond.
- [ ] **Mobile App:** Port the Streamlit prototype to React Native for iOS/Android access.

---

## Setup & Run

1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/yourusername/serenity-ai.git](https://github.com/yourusername/serenity-ai.git)
    ```
2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Add Keys:**
    Open `app.py` and paste your Google & Deepgram API keys.
4.  **Launch:**
    ```bash
    streamlit run app.py
    ```

---

## ⚠️ Disclaimer
*This project is for educational and research purposes only. It is not a replacement for professional medical advice, diagnosis, or treatment. If you are in crisis, please call your local emergency number immediately.*
