import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ai.interview import generate_questions_with_category, get_question_bank_templates, evaluate_answer_with_context
from utils.parser import extract_text_from_upload
from ui.styles import inject_style
from ai.confidence import estimate_confidence
from ai.scorecard import Scorecard

st.set_page_config(page_title="AI Interview Assistant", layout="wide")
inject_style()

st.title("AI Interview Assistant")
scorecard = st.session_state.get('scorecard') or Scorecard()
st.session_state['scorecard'] = scorecard
if 'questions' not in st.session_state:
    st.session_state['questions'] = []
if 'context' not in st.session_state:
    st.session_state['context'] = ""

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Interview", "Question Bank"])
    model = st.selectbox("LLM Provider", ["Google Gemini (default)"])
    question_category = st.selectbox("Question bank", ["Technical", "Behavioral", "Mixed"])
    num_q = st.slider("Number of questions", 5, 20, 10)
    st.markdown("---")
    st.write("Quick actions")
    if st.button("Clear questions"):
        st.session_state.pop('questions', None)
        st.session_state.pop('context', None)
        st.session_state.pop('evaluations', None)
        st.session_state.pop('scorecard', None)
        st.success("Questions and scorecard cleared")

col1, col2, col3 = st.columns([2, 4, 2])

with col1:
    st.header("Candidate Context")
    resume_file = st.file_uploader("Upload resume (txt or pdf)", type=["txt", "pdf"] )
    jd_file = st.file_uploader("Upload job description (txt or pdf)", type=["txt", "pdf"] )
    resume_text = extract_text_from_upload(resume_file)
    jd_text = extract_text_from_upload(jd_file)

    st.markdown("---")
    st.header("Text Interview")
    st.write("This application now runs a professional text-only interview flow.")
    st.write("Upload a resume or job description to generate interview questions tailored to the role, then answer them directly in text.")
    st.markdown("---")
    st.write("**Question bank categories:**")
    st.write("- Technical: core engineering, AI, ML, and system design questions")
    st.write("- Behavioral: communication, leadership, problem solving, and cultural fit questions")
    st.write("- Mixed: a balanced interview set with both technical and behavioral questions")

with col2:
    if mode == "Question Bank":
        st.header("Question Bank")
        st.write("Build a professional interview question set from category templates and custom questions.")
        bank_templates = get_question_bank_templates(question_category)
        # default to no pre-selected templates to avoid unexpected large loads
        selected_questions = st.multiselect(
            "Select questions from the bank",
            bank_templates,
            default=[]
        )
        custom_questions = st.text_area(
            "Add custom questions (one per line)",
            height=150,
        )
        if st.button("Load selected questions into interview"):
            custom_list = [q.strip() for q in custom_questions.splitlines() if q.strip()]
            loaded_questions = selected_questions + custom_list
            if not loaded_questions:
                st.warning("Select or add at least one question to load.")
            else:
                st.session_state['questions'] = loaded_questions[:num_q]
                st.session_state['context'] = ""
                st.session_state['question_category'] = question_category
                st.success(f"Loaded {len(st.session_state['questions'])} questions into interview mode.")
        st.markdown("---")
        st.subheader("Question bank preview")
        for sample_q in bank_templates[:10]:
            st.write(f"- {sample_q}")
    else:
        st.header("Generated Interview Questions")
        if st.button("Generate Questions"):
            if not resume_text and not jd_text:
                st.info("No resume or JD uploaded. Loading template questions from the selected question bank.")
            qs, context = generate_questions_with_category(resume_text, jd_text, question_category, n=num_q)
            if not qs:
                qs = get_question_bank_templates(question_category)[:num_q]
                context = ""
            st.session_state['questions'] = qs
            st.session_state['context'] = context
            st.session_state['question_category'] = question_category
            # option to persist RAG index to disk (rag_index)
            if st.checkbox("Persist RAG index to disk (rag_index)"):
                try:
                    from ai.rag import RAGStore
                    docs = []
                    if resume_text:
                        docs.append(resume_text)
                    if jd_text:
                        docs.append(jd_text)
                    rag = RAGStore(docs)
                    rag.persist('rag_index')
                    st.success('RAG index saved to rag_index.index')
                except Exception as e:
                    st.error(f"Failed to persist RAG index: {e}")

        if 'questions' in st.session_state:
            sample_questions = get_question_bank_templates(st.session_state.get('question_category', question_category))
            st.subheader("Sample question bank")
            for sample_q in sample_questions[:4]:
                st.write(f"- {sample_q}")
            st.markdown("---")

    if st.session_state.get('questions'):
        st.markdown("---")
        st.header("Interview Questions")
        if st.button("Auto-evaluate all answers"):
            scorecard = st.session_state.get('scorecard') or Scorecard()
            st.session_state['scorecard'] = scorecard
            evaluations = []
            with st.spinner("Evaluating answers..."):
                for i, q in enumerate(st.session_state['questions'], 1):
                    ans_key = f"ans_{i}"
                    ans = st.session_state.get(ans_key, "") or ""
                    if not ans.strip():
                        continue
                    try:
                        res = evaluate_answer_with_context(q, ans, st.session_state.get('context', ""))
                    except Exception as e:
                        res = {'score': None, 'feedback': f'Evaluation failed: {e}', 'strengths': '', 'weaknesses': '', 'improvement_suggestions': ''}
                    try:
                        from ai.confidence import estimate_and_calibrate
                        conf = estimate_and_calibrate(ans, evaluator_score=res.get('score'))
                        conf_score = conf.get('calibrated_confidence')
                    except Exception:
                        conf = estimate_confidence(ans)
                        conf_score = conf.get('confidence')
                    scorecard.add(i, q, ans, score=res.get('score'), feedback=res.get('feedback'), improvement_suggestions=res.get('improvement_suggestions'), confidence=conf_score)
                    evaluations.append({
                        'q_no': i,
                        'question': q,
                        'answer': ans,
                        'score': res.get('score'),
                        'feedback': res.get('feedback'),
                        'strengths': res.get('strengths'),
                        'weaknesses': res.get('weaknesses'),
                        'improvement_suggestions': res.get('improvement_suggestions'),
                        'confidence': conf_score,
                    })
            st.session_state['evaluations'] = evaluations
            st.success('Auto-evaluation complete and added to scorecard')

        for i, q in enumerate(st.session_state['questions'], 1):
            st.markdown(f"<div class=\"question\">\n**Q{i}. {q}**\n</div>", unsafe_allow_html=True)
            ans = st.text_area(f"Your answer for Q{i}", value=st.session_state.get(f"ans_{i}", ""), key=f"ans_{i}", height=150)
            cols = st.columns([1,1,1])
            if cols[0].button(f"Evaluate Q{i}", key=f"eval_{i}"):
                if not ans or ans.strip() == "":
                    st.warning("Please enter an answer first")
                else:
                    context = st.session_state.get('context', "")
                    res = evaluate_answer_with_context(q, ans, context)
                    if isinstance(res, dict):
                        score = res.get('score')
                        feedback = res.get('feedback', '')
                        strengths = res.get('strengths', '')
                        weaknesses = res.get('weaknesses', '')
                        improvements = res.get('improvement_suggestions', '')
                        
                        col_eval1, col_eval2 = st.columns(2)
                        with col_eval1:
                            if score is not None:
                                st.metric("Score", f"{score}/10")
                            else:
                                st.warning("Score unavailable: LLM evaluation failed or quota was exceeded.")
                            st.write(f"**Feedback:**\n{feedback}")
                            if improvements:
                                st.write(f"**Improvement suggestions:**\n{improvements}")
                        with col_eval2:
                            st.write(f"**Strengths:**\n{strengths}")
                            st.write(f"**Weaknesses:**\n{weaknesses}")
                        
                        # If score unavailable from LLM, use local fallback
                        if score is None:
                            try:
                                from ai.interview import local_evaluate
                                local_res = local_evaluate(q, ans)
                                score = local_res.get('score')
                                feedback = local_res.get('feedback')
                                strengths = local_res.get('strengths')
                                weaknesses = local_res.get('weaknesses')
                                improvements = local_res.get('improvement_suggestions')
                                fallback_note = "(Local heuristic used)"
                            except Exception:
                                fallback_note = "(Evaluation unavailable)"
                        else:
                            fallback_note = ""

                        conf = estimate_confidence(ans)
                        conf_val = conf.get('confidence') if isinstance(conf, dict) else None
                        scorecard.add(i, q, ans, score=score, feedback=feedback + " " + fallback_note, improvement_suggestions=improvements, confidence=conf_val)
                        st.success('Evaluation added to scorecard')
                    else:
                        st.error("Failed to evaluate answer")
            if cols[1].button(f"Estimate Confidence Q{i}", key=f"conf_{i}"):
                if not ans or ans.strip() == "":
                    st.warning("Please enter an answer first")
                else:
                    conf = estimate_confidence(ans)
                    if isinstance(conf, dict):
                        confidence_score = conf.get('confidence')
                        cues = conf.get('cues', '')
                        explanation = conf.get('explanation', '')
                        if confidence_score is not None:
                            st.metric("Confidence Level", f"{int(confidence_score * 100)}%")
                        st.write(f"**Verbal Cues:**\n{cues}")
                        st.write(f"**Analysis:**\n{explanation}")
                    else:
                        st.error("Failed to estimate confidence")

with col3:
    st.header("Interview Summary")
    if scorecard.rows:
        answered = len([r for r in scorecard.rows if r.get('answer') and r.get('answer').strip()])
        scores = [r['score'] for r in scorecard.rows if isinstance(r.get('score'), int)]
        average_score = sum(scores) / len(scores) if scores else None
        st.metric("Questions answered", answered)
        if average_score is not None:
            st.metric("Average score", f"{average_score:.1f}/10")
        st.markdown("---")
        st.subheader("Scorecard details")
        for r in scorecard.rows:
            score_text = f"{r['score']}" if r.get('score') is not None else "N/A"
            confidence_text = f"{r['confidence']}" if r.get('confidence') is not None else "N/A"
            st.write(f"**Q{r['q_no']}** - score={score_text} confidence={confidence_text}")
            st.write(f"- Feedback: {r['feedback']}")
            if r.get('improvement_suggestions'):
                st.write(f"- Suggestions: {r['improvement_suggestions']}")
        csv = scorecard.to_csv()
        st.download_button("Download Scorecard CSV", csv, file_name="scorecard.csv", mime='text/csv')
    else:
        st.info("No evaluations yet - answer questions or use Auto-evaluate to build a scorecard.")

st.markdown("---")
st.caption("Prototype: adapt models, keys, and UI for production.")
