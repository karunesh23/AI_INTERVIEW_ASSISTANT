from .llm import _call_llm


def estimate_confidence(answer_text: str) -> dict:
    """Ask the LLM to estimate confidence and list verbal cues.
    Returns: {confidence: float (0-1), cues: str, explanation: str}
    """
    prompt = (
        f"Estimate the candidate's speaking confidence from the following answer. "
        f"Return JSON with fields: confidence (float 0-1), cues (short list of verbal/nonverbal cues), explanation (1-2 sentences).\n\nAnswer:\n{answer_text}\n\n"
        "Return ONLY a JSON object. No markdown, no code blocks."
    )
    try:
        out = _call_llm(prompt, max_tokens=300)
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
            if 'confidence' in obj and obj['confidence'] is not None:
                try:
                    conf = float(obj['confidence'])
                    obj['confidence'] = min(1.0, max(0.0, conf))
                except:
                    obj['confidence'] = None
            return obj
    except Exception:
        pass
    return {"confidence": None, "cues": "", "explanation": "Unable to estimate confidence"}


def calibrate_confidence(llm_confidence: float | None, mapping_score: float | None = None, evaluator_score: float | None = None) -> float | None:
    """Simple calibration: combine LLM verbal-confidence with mapping quality and evaluator score.

    - llm_confidence: 0-1 or None
    - mapping_score: similarity 0-1 or None
    - evaluator_score: 0-10 or None

    Returns combined confidence in 0-1 or None if insufficient data.
    """
    try:
        parts = []
        weights = []
        if llm_confidence is not None:
            parts.append(float(llm_confidence))
            weights.append(0.6)
        if mapping_score is not None:
            # mapping_score from similarity model (cosine) or difflib ratio (0-1)
            parts.append(float(mapping_score))
            weights.append(0.2)
        if evaluator_score is not None:
            # scale evaluator_score (0-10) to 0-1
            parts.append(float(evaluator_score) / 10.0)
            weights.append(0.2)

        if not parts:
            return None

        # weighted average
        total_weight = sum(weights)
        weighted = sum(p * w for p, w in zip(parts, weights)) / total_weight
        # simple calibration curve: shrink extreme values slightly
        calibrated = 0.95 * weighted + 0.025
        return min(1.0, max(0.0, calibrated))
    except Exception:
        return None


def estimate_and_calibrate(answer_text: str, mapping_score: float | None = None, evaluator_score: float | None = None) -> dict:
    """Convenience wrapper: calls `estimate_confidence` and then `calibrate_confidence`."""
    res = estimate_confidence(answer_text)
    llm_conf = res.get('confidence')
    combined = calibrate_confidence(llm_conf, mapping_score=mapping_score, evaluator_score=evaluator_score)
    res['calibrated_confidence'] = combined
    return res

def estimate_confidence_with_face(answer_text: str, image_path: str = None) -> dict:
    """Combine LLM-based verbal confidence and visual confidence (via DeepFace) if image_path provided."""
    res = estimate_confidence(answer_text)
    visual = None
    if image_path:
        try:
            from .face_confidence import analyze_face
            visual = analyze_face(image_path)
            res['visual'] = visual
            # combine a simple aggregate: average of LLM confidence and visual avg_confidence
            try:
                v = visual['summary'].get('avg_confidence', 0.0)
            except Exception:
                v = 0.0
            try:
                l = float(res.get('confidence') or 0.0)
            except Exception:
                l = 0.0
            if l is not None:
                res['combined_confidence'] = (l + v) / 2.0
            else:
                res['combined_confidence'] = v
        except Exception as e:
            res['visual_error'] = str(e)
    return res
