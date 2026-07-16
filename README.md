# AI Interview Assistant

Prototype AI Interview Assistant using Streamlit, OpenAI/Gemini, Whisper, Sentence-Transformers, and FAISS.

Features
- Resume upload
- JD upload
- AI generates interview questions
- Voice interview (audio upload / Whisper)
- AI evaluates answers and gives feedback
- Scorecard export

Quick start
1. Create a virtual env and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

2. Set API keys (if using OpenAI):

```bash
export OPENAI_API_KEY="sk-..."  # Windows: setx OPENAI_API_KEY "sk-..."
```

3. Run the app:

```bash
streamlit run app.py
```

Notes
- This is a prototype. Replace model calls with Gemini or your preferred LLM by updating `ai/llm.py`.
- Whisper transcription falls back to local `whisper` if no OpenAI audio key is available.
