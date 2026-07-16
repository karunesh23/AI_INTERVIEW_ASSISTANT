from .llm import generate_questions, evaluate_answer
from .rag import RAGStore
from typing import List, Dict
import difflib


def map_transcript_to_questions(transcript: Dict, questions: List[str], window_size: int = 3) -> List[Dict]:
    """Map transcribed segments to each question.

    transcript: {'text': str, 'segments': [{'start':, 'end':, 'text':}, ...]}
    Returns list of mappings per question:
    [{ 'q_no': int, 'question': str, 'matched_text': str, 'start': float, 'end': float, 'score': float }]
    """
    segments = transcript.get('segments') or []
    full_text = transcript.get('text', '')

    # If no fine-grained segments, fallback to full_text as single segment
    if not segments:
        segments = [{"start": 0.0, "end": 0.0, "text": full_text}]

    # Build candidate windows by joining up to window_size consecutive segments
    windows = []
    n = len(segments)
    for i in range(n):
        for w in range(1, window_size + 1):
            end_idx = min(n, i + w)
            segs = segments[i:end_idx]
            text = " ".join(s['text'] for s in segs)
            start = segs[0]['start']
            end = segs[-1]['end']
            windows.append({"start": start, "end": end, "text": text, "i": i, "j": end_idx - 1})

    mappings = []

    # Try to use sentence-transformers if available for better similarity
    use_emb = False
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('all-MiniLM-L6-v2')
        use_emb = True
    except Exception:
        model = None

    if use_emb:
        # Compute embeddings for windows and questions
        window_texts = [w['text'] for w in windows]
        q_embeddings = model.encode(questions, convert_to_tensor=True)
        w_embeddings = model.encode(window_texts, convert_to_tensor=True)

        for qi, q in enumerate(questions):
            sims = util.cos_sim(q_embeddings[qi], w_embeddings)[0].cpu().tolist()
            best_idx = int(max(range(len(sims)), key=lambda k: sims[k]))
            score = float(sims[best_idx])
            w = windows[best_idx]
            mappings.append({
                'q_no': qi + 1,
                'question': q,
                'matched_text': w['text'],
                'start': w['start'],
                'end': w['end'],
                'score': score,
            })
    else:
        # Fallback to difflib SequenceMatcher ratio on lowercase text
        for qi, q in enumerate(questions):
            qnorm = q.lower()
            best_score = 0.0
            best_w = None
            for w in windows:
                # compute ratio
                try:
                    s = difflib.SequenceMatcher(None, qnorm, w['text'].lower())
                    r = s.ratio()
                except Exception:
                    r = 0.0
                if r > best_score:
                    best_score = r
                    best_w = w
            if best_w is None:
                matched_text = ""
                start = 0.0
                end = 0.0
                score = 0.0
            else:
                matched_text = best_w['text']
                start = best_w['start']
                end = best_w['end']
                score = best_score

            mappings.append({
                'q_no': qi + 1,
                'question': q,
                'matched_text': matched_text,
                'start': start,
                'end': end,
                'score': score,
            })

    return mappings


def prepare_documents(resume_text: str, jd_text: str):
    # naive splitting
    docs = []
    if resume_text:
        docs.append(resume_text)
    if jd_text:
        docs.append(jd_text)
    return docs


def generate_questions_rag(resume_text: str, jd_text: str, n: int = 10):
    docs = prepare_documents(resume_text, jd_text)
    # try to build a RAG store; if embeddings/faiss not available, fall back gracefully
    context = "\n\n".join(docs)
    try:
        rag = RAGStore(docs)
        # create a context string from top-k passages
        context_pieces = []
        if jd_text:
            results = rag.query(jd_text, k=4)
            context_pieces = [r[0] for r in results]
        if context_pieces:
            context = "\n\n".join(context_pieces)
    except Exception:
        # fallback to using full docs as context (no RAG)
        context = "\n\n".join(docs)
    return generate_questions(resume_text, jd_text, n=n), context


def get_question_bank_templates(category: str = "Technical") -> list:
    category = category.lower().strip()
    templates = {
        'technical': [
            "Explain a machine learning project you built end-to-end and the key challenges you solved.",
            "How do you choose between a decision tree, random forest, and XGBoost for a classification task?",
            "Describe the process of tuning a neural network and avoiding overfitting.",
            "What is RAG and how would you apply it in a product with a document database?",
            "Explain how you would design a scalable data pipeline for real-time model serving.",
            "How do you evaluate model performance when the dataset is imbalanced?",
            "What steps do you take to prevent data leakage in a machine learning pipeline?",
            "Compare batch training and online learning for production model deployment.",
            "Describe how transformers work and why they are used in NLP tasks.",
            "Explain the difference between precision, recall, and F1 score.",
            "How would you explain the bias-variance tradeoff to a non-technical stakeholder?",
            "What are embeddings, and how are they used in search or recommendation systems?",
            "Describe a situation where you used feature engineering to improve model accuracy.",
            "Explain how you would deploy a model to a cloud service with monitoring.",
            "What is transfer learning, and when would you use it?",
            "How do you choose a loss function for a regression versus classification problem?",
            "Describe the role of regularization in neural networks.",
            "What is the difference between supervised, unsupervised, and reinforcement learning?",
            "Explain how you would handle missing values in a dataset.",
            "What are the main considerations when building a recommendation system?",
            "How do you explain the performance of a model to business stakeholders?"
        ],
        'behavioral': [
            "Describe a time when you had to influence a team member to adopt a better technical solution.",
            "How do you handle feedback or criticism on your code or design choices?",
            "Tell me about a situation where you had to learn a new technology quickly to complete a project.",
            "Explain how you prioritize tasks when multiple project deadlines overlap.",
            "How do you communicate complex technical ideas to non-technical stakeholders?",
            "Describe a challenging project and how you kept your team motivated.",
            "How do you manage stress when working under tight deadlines?",
            "Tell me about a time you made a mistake and how you fixed it.",
            "Explain how you approach collaboration with cross-functional teams.",
            "Describe a time when you had to solve a conflict within your team.",
            "How do you set and measure your own performance goals?",
            "Explain a time when you had to persuade others to support your idea.",
            "Describe how you stay organized when handling multiple priorities.",
            "Tell me about a time you simplified a complex task for others.",
            "How do you ensure clear communication on a remote or distributed team?",
            "Describe a situation where you had to adapt to changing project requirements.",
            "How do you support mentorship and knowledge sharing in your team?",
            "Describe a time when you demonstrated leadership without a formal title.",
            "How do you build trust with stakeholders and partners?",
            "Explain how you handle difficult conversations with colleagues.",
            "Describe a project where you took ownership of the outcome."
        ],
        'mixed': [
            "Describe a technical challenge you solved and how you collaborated with others during the process.",
            "Explain a time you improved a product or process through data or automation.",
            "How do you balance code quality and delivery speed in a product-focused environment?",
            "Describe a project where you used AI or machine learning to solve a business problem.",
            "How do you approach debugging a production issue under pressure?",
            "Describe a situation when you needed to communicate technical risk to a stakeholder.",
            "Explain how you handled a project where requirements changed rapidly.",
            "Describe a time when you helped another team member grow their technical skills.",
            "How do you decide when to simplify a solution rather than create a complex design?",
            "Explain a time you had to translate user feedback into engineering work.",
            "Describe how you prioritize roadmap features when resources are limited.",
            "How do you ensure that technical decisions align with business goals?",
            "Explain a time when you built a solution that improved user experience.",
            "Describe a time when you used data to influence product strategy.",
            "How do you keep learning new technologies while working on delivery?",
            "Explain how you handled failure in a project and what you learned.",
            "Describe how you balance stakeholder requests with technical debt management.",
            "How do you ensure your team stays aligned on project priorities?",
            "Explain an example where you used analytics to make a better decision.",
            "Describe a time when you had to take initiative and drive a project forward.",
            "How do you foster a culture of quality and continuous improvement?"
        ]
    }
    return templates.get(category, templates['technical'])


def generate_questions_with_category(resume_text: str, jd_text: str, category: str = "Technical", n: int = 10):
    """Generate interview questions using a selected category bank."""
    category = category.lower().strip()
    prompt_category = {
        'technical': "Technical interview questions focusing on machine learning, AI, data science, programming, and system design.",
        'behavioral': "Behavioral interview questions focusing on communication, leadership, teamwork, problem-solving, and culture fit.",
        'mixed': "A balanced interview set with both technical and behavioral questions suitable for a professional candidate assessment."
    }.get(category, "Technical interview questions focusing on machine learning, AI, data science, programming, and system design.")

    docs = prepare_documents(resume_text, jd_text)
    context = "\n\n".join(docs)
    try:
        rag = RAGStore(docs)
        context_pieces = []
        if jd_text:
            results = rag.query(jd_text, k=4)
            context_pieces = [r[0] for r in results]
        if context_pieces:
            context = "\n\n".join(context_pieces)
    except Exception:
        context = "\n\n".join(docs)

    # ask the LLM to generate questions with category guidance
    prompt = (
        f"You are an expert interview question generator. Create exactly {n} well-formed interview questions. "
        f"Focus on: {prompt_category}\n\n"
        f"Resume context:\n{resume_text[:500] if resume_text else 'Not provided'}\n\n"
        f"Job description context:\n{jd_text[:500] if jd_text else 'Not provided'}\n\n"
        "Return ONLY a JSON array of questions. No markdown."
    )

    try:
        import json
        from .llm import _call_llm
        out = _call_llm(prompt, max_tokens=1500)
        out_clean = out.strip()
        if out_clean.startswith('```'):
            out_clean = out_clean.split('```')[1]
            if out_clean.startswith('json'):
                out_clean = out_clean[4:]
        start = out_clean.find('[')
        end = out_clean.rfind(']') + 1
        if start >= 0 and end > start:
            arr = json.loads(out_clean[start:end])
            questions = []
            for item in arr:
                if isinstance(item, str):
                    questions.append(item)
                elif isinstance(item, dict):
                    questions.append(item.get('question', str(item)))
                else:
                    questions.append(str(item))
            # shuffle to add some variability
            try:
                import random
                random.shuffle(questions)
            except Exception:
                pass
            return questions[:n], context
    except Exception:
        pass

    # fallback to generic question generation if LLM response parsing fails
    try:
        return generate_questions(resume_text, jd_text, n=n), context
    except Exception:
        # final fallback: use templates but shuffle for variety
        templ = get_question_bank_templates(category)[:]
        try:
            import random
            random.shuffle(templ)
        except Exception:
            pass
        return templ[:n], context


def local_evaluate(question: str, answer: str) -> Dict:
    """Lightweight local evaluation fallback when LLM is unavailable.

    Returns: {score:int, feedback:str, strengths:str, weaknesses:str, improvement_suggestions:str}
    """
    # compute simple similarity-based score
    qtxt = question.lower()
    atxt = answer.lower()
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('all-MiniLM-L6-v2')
        q_emb = model.encode(qtxt, convert_to_tensor=True)
        a_emb = model.encode(atxt, convert_to_tensor=True)
        sim = float(util.cos_sim(q_emb, a_emb).item())
    except Exception:
        # fallback to difflib ratio
        try:
            import difflib
            sim = difflib.SequenceMatcher(None, qtxt, atxt).ratio()
        except Exception:
            sim = 0.0

    # scale to 0-10
    score = int(round(max(0.0, min(1.0, sim)) * 10))

    feedback = f"Local heuristic score based on text similarity ({sim:.2f})."
    strengths = "Answer mentions relevant keywords." if sim > 0.3 else "Needs more relevant detail."
    weaknesses = "May lack depth or examples." if sim < 0.6 else ""
    improvement = "Add concrete examples, metrics, and specific steps to improve the answer."

    return {
        'score': score,
        'feedback': feedback,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'improvement_suggestions': improvement,
    }


def evaluate_answer_with_context(question: str, answer: str, context: str = ""):
    return evaluate_answer(question, answer, context)
