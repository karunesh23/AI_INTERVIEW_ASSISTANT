# 🤖 AI Interview Assistant

### An Intelligent Interview Preparation Platform Powered by Generative AI

AI Interview Assistant is a Generative AI-based web application that simulates real technical and behavioral interviews. It generates personalized interview questions based on a candidate's resume and job description, evaluates answers using Large Language Models (LLMs), provides detailed feedback with confidence scores, and helps candidates improve their interview performance.

---

## 🚀 Features

### 📄 Resume Analysis
- Upload Resume (PDF)
- Extract candidate skills
- Identify projects and experience
- Generate interview questions based on resume

### 💼 Job Description Analysis
- Upload or paste Job Description
- Extract required skills
- Match resume with JD
- Generate role-specific interview questions

### 🤖 AI Interview Generation
- Technical Interview
- Behavioral Interview
- HR Interview
- Mixed Interview Mode

### 🎯 AI Answer Evaluation
- Evaluate candidate answers
- Generate detailed feedback
- Suggest improvements
- Explain correct concepts

### 📊 Confidence Score
- AI-generated confidence score
- Communication assessment
- Technical accuracy analysis

### 📚 Retrieval-Augmented Generation (RAG)
- Uses document embeddings
- Retrieves relevant resume/JD information
- Generates context-aware interview questions

### 🧠 Personalized Feedback
- Strengths
- Weaknesses
- Areas to Improve
- Final Recommendation

---

# 🛠 Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Frontend | Streamlit |
| LLM | Google Gemini |
| Framework | LangChain |
| Embeddings | Google Generative AI Embeddings |
| RAG | Retrieval-Augmented Generation |
| PDF Parsing | PyPDF |
| Vector Search | In-Memory Embeddings |
| Prompt Engineering | LangChain Prompt Templates |

---

# 📂 Project Structure

```
AI_INTERVIEW_ASSISTANT
│
├── ai/
│   ├── chunker.py
│   ├── confidence.py
│   ├── embeddings.py
│   ├── interview.py
│   ├── llm.py
│   ├── rag.py
│   └── scorecard.py
│
├── ui/
│   └── styles.py
│
├── utils/
│   └── parser.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🔑 Environment Variables

Create a `.env` file

```text
GOOGLE_API_KEY=YOUR_API_KEY
```

---

# ▶ Run the Application

```bash
streamlit run app.py
```

---

# 🧠 Application Workflow

```
Resume Upload
        │
        ▼
Resume Parsing
        │
        ▼
Job Description Analysis
        │
        ▼
Document Chunking
        │
        ▼
Embedding Generation
        │
        ▼
RAG Retrieval
        │
        ▼
Gemini LLM
        │
        ▼
Interview Question Generation
        │
        ▼
Candidate Answer
        │
        ▼
Answer Evaluation
        │
        ▼
Confidence Score
        │
        ▼
Personalized Feedback
```

---

# 💡 Key Highlights

- AI-powered interview simulation
- Resume-aware interview generation
- Job description matching
- LLM-based answer evaluation
- Confidence score prediction
- RAG-based contextual question generation
- Personalized interview feedback
- Interactive Streamlit UI

---

# 🎯 Use Cases

- Campus Placement Preparation
- AI/ML Interview Practice
- Software Developer Interviews
- HR Interview Preparation
- Resume Screening
- Technical Assessment
- Self Evaluation

---

# 📈 Future Improvements

- Voice-based Interview
- Video Interview Analysis
- Facial Emotion Detection
- Real-time AI Interviewer
- ATS Resume Score
- Interview History Dashboard
- Export Interview Report (PDF)
- Multi-language Support
- Authentication & User Profiles
- Leaderboard & Progress Tracking

---

# 👨‍💻 Author

## Karunesh Bansal

**AI | Machine Learning | Generative AI Enthusiast**

🔗 **GitHub:** https://github.com/karunesh23

🔗 **LinkedIn:** https://www.linkedin.com/in/karunesh-bansal

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

---

## 📜 License

This project is licensed under the MIT License.