import os
import re
from typing import List, Dict

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    import google.genai as genai
    genai_available = GEMINI_API_KEY is not None
except Exception as e:
    genai = None
    genai_available = False


def _call_llm(prompt: str, max_tokens: int = 512) -> str:
    """Simple LLM call wrapper. Uses Google Gemini if available, otherwise raises."""
    if not genai_available or not GEMINI_API_KEY:
        raise RuntimeError("Google Genai SDK not available or GEMINI_API_KEY not set")
    
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {str(e)}")


def generate_questions(resume_text: str, jd_text: str, n: int = 10) -> List[str]:
    prompt = (
        "You are an expert ML, DL, AI, and Python interview question generator.\n"
        "Generate exactly " + str(n) + " specific, well-formed technical interview questions.\n\n"
        "Topics to focus on (based on resume and JD):\n"
        "- Machine Learning (XGBoost, Random Forest, SVM, Clustering, etc.)\n"
        "- Deep Learning (Neural Networks, CNNs, RNNs, Transformers, etc.)\n"
        "- Generative AI (LLMs, RAG, Prompting, Fine-tuning, etc.)\n"
        "- NLP (Tokenization, Embeddings, Transformers, Text Classification, etc.)\n"
        "- Python (Data structures, OOP, async, decorators, etc.)\n"
        "- Data Science (Data Leakage, Feature Engineering, Model Evaluation, etc.)\n\n"
        "Resume context:\n" + (resume_text[:500] if resume_text else "Not provided") + "\n\n"
        "Job Description context:\n" + (jd_text[:500] if jd_text else "Not provided") + "\n\n"
        "Return ONLY a JSON array with exactly " + str(n) + " questions. Each question must be:\n"
        "- Clear and specific (NOT vague or generic)\n"
        "- A single line (no newlines in the question)\n"
        "- Focused on technical concepts\n"
        "- Real interview-style questions\n\n"
        "Format: [\"Question 1?\", \"Question 2?\", \"Question 3?\", ...]\n\n"
        "Return ONLY the JSON array. No markdown, no code blocks, no extra text."
    )
    try:
        out = _call_llm(prompt, max_tokens=1500)
        import json
        try:
            # Clean up the response - remove markdown code blocks if present
            out_clean = out.strip()
            if out_clean.startswith("```"):
                out_clean = out_clean.split("```")[1]
                if out_clean.startswith("json"):
                    out_clean = out_clean[4:]
            
            # Find JSON array
            start = out_clean.find('[')
            end = out_clean.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = out_clean[start:end]
                arr = json.loads(json_str)
                
                # Handle both array of strings and array of objects
                questions = []
                for item in arr:
                    if isinstance(item, str):
                        questions.append(item)
                    elif isinstance(item, dict):
                        questions.append(item.get('question', str(item)))
                    else:
                        questions.append(str(item))
                return questions[:n]
        except Exception:
            pass
    except Exception:
        pass
    
    # Fallback: extract questions line by line
    try:
        questions = []
        lines = out.split('\n')
        for line in lines:
            line = line.strip()
            # Skip empty lines, markdown markers, and non-question lines
            if not line or line.startswith('```') or line.startswith('#'):
                continue
            # Remove common prefixes like "- ", "1. ", etc.
            if line[0] in ['-', '*', '•']:
                line = line[1:].strip()
            if line and len(line) > 10 and ('?' in line or 'what' in line.lower() or 'how' in line.lower() or 'explain' in line.lower()):
                # Remove numbering if present
                if line[0].isdigit():
                    line = line.split('.', 1)[1].strip() if '.' in line else line[1:].strip()
                questions.append(line)
        
        if questions:
            return questions[:n]
    except Exception:
        pass
    
    # Last resort: return generic questions
    return [
        "What is XGBoost and how does it differ from traditional gradient boosting?",
        "Explain the concept of data leakage and provide an example.",
        "What are the key differences between CNNs and RNNs?",
        "How would you handle imbalanced datasets in a classification problem?",
        "What is transfer learning and when would you use it?",
        "Explain RAG (Retrieval Augmented Generation) and its applications.",
        "What are transformers and how do they work?",
        "How do you evaluate NLP models?",
        "What is feature engineering and why is it important?",
        "Explain the difference between overfitting and underfitting."
    ][:n]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _keyword_coverage(question: str, answer: str) -> float:
    stopwords = {
        'what', 'when', 'where', 'how', 'why', 'which', 'and', 'or', 'the', 'a', 'an', 'in', 'on', 'for', 'to',
        'of', 'is', 'are', 'do', 'does', 'did', 'with', 'as', 'by', 'from', 'that', 'this', 'it', 'its',
        'your', 'you', 'i', 'we', 'be', 'was', 'were', 'have', 'has', 'had', 'but', 'not', 'can', 'could', 'will', 'would'
    }
    q_tokens = [t for t in re.findall(r"\b\w+\b", question.lower()) if len(t) > 3 and t not in stopwords]
    if not q_tokens:
        return 0.0
    normalized_answer = _normalize_text(answer)
    hits = sum(1 for token in set(q_tokens) if token in normalized_answer)
    return min(1.0, hits / len(set(q_tokens)))


def _local_evaluate_answer(question: str, answer: str) -> Dict:
    answer_text = answer.strip()
    if not answer_text:
        return {
            'score': 0,
            'feedback': 'No answer provided. Please answer the question directly.',
            'strengths': '',
            'weaknesses': 'The response is empty.',
            'improvement_suggestions': 'Provide a concise and structured answer that addresses the question clearly.'
        }

    normalized_answer = _normalize_text(answer_text)
    if any(neg in normalized_answer for neg in ['i don\'t know', 'not sure', 'cannot answer', 'no idea']):
        base_score = 2
    else:
        base_score = min(9, max(2, len(normalized_answer) // 40))

    keyword_score = _keyword_coverage(question, answer_text)
    score = int(min(10, max(1, round(base_score * 0.5 + keyword_score * 10 * 0.5))))

    strengths = []
    weaknesses = []
    suggestions = []

    if len(normalized_answer) > 120:
        strengths.append('Answer is detailed and shows understanding.')
    else:
        weaknesses.append('Answer is brief and could use more detail.')
        suggestions.append('Expand your answer with examples, concepts, or results.')

    if keyword_score > 0.6:
        strengths.append('Answer includes relevant terms from the question.')
    else:
        weaknesses.append('Missing some key technical terms or concepts from the question.')
        suggestions.append('Include more direct references to the main concepts asked by the question.')

    if 'example' not in normalized_answer and score < 7:
        suggestions.append('Use an example or real scenario to make the answer stronger.')

    return {
        'score': score,
        'feedback': 'Local fallback evaluation used. Score is based on answer length and key term coverage.',
        'strengths': ' '.join(strengths).strip() or 'The answer has a clear structure.',
        'weaknesses': ' '.join(weaknesses).strip() or 'The response may need more detail for full marks.',
        'improvement_suggestions': ' '.join(suggestions).strip() or 'Provide more technical detail and use examples where possible.'
    }


def evaluate_answer(question: str, answer: str, context: str = "") -> Dict:
    prompt = (
        "You are an experienced interviewer and evaluator.\n"
        "Given the question, the candidate's answer, and optional context, provide an objective evaluation as JSON.\n"
        "Score the answer from 0-10 (integer). Provide concise 'feedback', 'strengths', 'weaknesses', and 'improvement_suggestions'.\n\n"
        "Question:\n" + question + "\n\nAnswer:\n" + answer + "\n\nContext:\n" + context + "\n\n"
        "Return ONLY a JSON object with keys: score (integer 0-10), feedback (string), strengths (string), weaknesses (string), improvement_suggestions (string)."
    )
    try:
        out = _call_llm(prompt, max_tokens=400)
        import json
        out_clean = out.strip()
        if out_clean.startswith("```"):
            out_clean = out_clean.split("```")[1]
            if out_clean.startswith("json"):
                out_clean = out_clean[4:]
        
        start = out_clean.find('{')
        end = out_clean.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = out_clean[start:end]
            obj = json.loads(json_str)
            if 'score' in obj and obj['score'] is not None:
                try:
                    obj['score'] = int(obj['score'])
                except Exception:
                    obj['score'] = None
            obj.setdefault('feedback', '')
            obj.setdefault('strengths', '')
            obj.setdefault('weaknesses', '')
            obj.setdefault('improvement_suggestions', '')
            return obj
    except Exception:
        pass

    # Fallback when the LLM is unavailable or parsing fails.
    return _local_evaluate_answer(question, answer)


def transcribe_audio_local(path: str) -> str:
    try:
        import whisper
        model = whisper.load_model("small")
        result = model.transcribe(path)
        if not isinstance(result, dict):
            raise ValueError("Whisper returned unexpected transcription format")
        text = result.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Whisper returned empty transcription")
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Local Whisper transcription failed: {str(e)}")


def transcribe_audio_gemini(path: str) -> str:
    if not genai_available or not GEMINI_API_KEY:
        raise RuntimeError("Google Genai SDK not available or GEMINI_API_KEY not set")
    try:
        import mimetypes
        client = genai.Client(api_key=GEMINI_API_KEY)

        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type or not mime_type.startswith("audio"):
            mime_type = "audio/wav"

        with open(path, "rb") as audio_file:
            print(f"Uploading audio file to Gemini (path: {path}, mime_type: {mime_type})...")
            file_response = client.files.upload(
                file=audio_file,
                config={"mime_type": mime_type, "display_name": os.path.basename(path)},
            )

        print(f"File uploaded successfully. Processing transcription...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                file_response,
                "Transcribe this audio file. Return ONLY the transcribed text, nothing else."
            ]
        )

        transcribed_text = response.text.strip() if response.text else ""
        if not transcribed_text:
            raise ValueError("Gemini returned empty transcription")
        return transcribed_text
    except Exception as e:
        raise RuntimeError(f"Gemini transcription failed: {str(e)}")


def transcribe_audio(path: str) -> str:
    """Transcribe audio file using Gemini API first, fallback to Whisper."""
    errors = []
    
    # Try Gemini first (preferred for better results)
    if genai_available and GEMINI_API_KEY:
        try:
            print("Attempting Gemini transcription...")
            return transcribe_audio_gemini(path)
        except Exception as e:
            errors.append(f"Gemini: {str(e)}")
            print(f"Gemini failed: {e}")
    
    # Fallback to Whisper
    try:
        print("Falling back to Whisper transcription...")
        return transcribe_audio_local(path)
    except Exception as e:
        errors.append(f"Whisper: {str(e)}")
        print(f"Whisper failed: {e}")
    
    # If both failed
    raise RuntimeError(f"All transcription methods failed:\n" + "\n".join(errors))


def transcribe_audio_struct(path: str) -> dict:
    """Return structured transcription: {'text': full_text, 'segments': [{'start':float,'end':float,'text':str}, ...]}.
    Tries Gemini (no timestamps), otherwise local Whisper which provides segments.
    """
    # Try Gemini first (no timestamps currently)
    if genai_available and GEMINI_API_KEY:
        try:
            text = transcribe_audio_gemini(path)
            return {"text": text, "segments": []}
        except Exception:
            pass

    # Fallback to local Whisper with segments
    try:
        import whisper
        model = whisper.load_model("small")
        result = model.transcribe(path)
        if not isinstance(result, dict):
            raise ValueError("Whisper returned unexpected transcription format")

        full_text = result.get("text", "").strip()
        segments_raw = result.get("segments") or []
        segments = []
        for seg in segments_raw:
            # Whisper segments typically have 'start', 'end', 'text'
            start = float(seg.get('start', 0.0))
            end = float(seg.get('end', start))
            text = seg.get('text', '').strip()
            if text:
                segments.append({"start": start, "end": end, "text": text})

        return {"text": full_text, "segments": segments}
    except Exception as e:
        raise RuntimeError(f"Structured local transcription failed: {str(e)}")
