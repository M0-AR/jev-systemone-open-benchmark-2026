"""System-One wire format: POST /v1/systemone compatible locally.

Request:  {"model": str, "state": str|dict|list, "questions": {qid: q}}
  choice: {"type":"choice","instructions":str,"criteria":{opt:desc}}
  score:  {"type":"score","instructions":str,"criteria":{level:desc}} (2-10 levels)
  noul:   {"type":"noul","instructions":str}
Response: {"model": str, "answers": {qid: answer}, "usage": {...}}
  choice -> {"type":"choice","choice":opt,"confidence":p,"probabilities":{...}}
  noul   -> {"type":"noul","noul":p}
  score  -> {"type":"score","score":float,"distribution":{...},"confidence":p}

This lets any Jev SDK run against a local proxy by changing base_url only,
which is exactly the drop-in claim (Kev/Laya/Clef) this repo verifies.
"""
import time


def _probs_to_choice_answer(q, probs):
    opts = list(q["criteria"].keys())
    best = opts[int(probs.argmax())]
    return {
        "type": "choice",
        "choice": best,
        "confidence": float(probs.max()),
        "probabilities": {o: float(p) for o, p in zip(opts, probs)},
    }


def _p_to_noul_answer(p):
    return {"type": "noul", "noul": float(p)}


def _probs_to_score_answer(q, probs):
    levels = list(q["criteria"].keys())
    idx = probs.argmax()
    # fractional score: probability-weighted rank, rescaled to level range
    score = float((probs * range(len(levels))).sum())
    return {
        "type": "score",
        "score": score,
        "distribution": {l: float(p) for l, p in zip(levels, probs)},
        "confidence": float(probs.max()),
    }


class SystemOneRequest(dict):
    pass


def systemone_local_predict(model_name, state, questions, predictor, t0=None):
    """predictor: fn(question_dict) -> prob vector aligned to options.

    For noul questions options are ["no","yes"] and p = probs[1].
    """
    t0 = t0 or time.perf_counter()
    answers = {}
    n_input_tokens = len(str(state).split())
    for qid, q in questions.items():
        probs = predictor(q)
        if q["type"] == "choice":
            answers[qid] = _probs_to_choice_answer(q, probs)
        elif q["type"] == "noul":
            answers[qid] = _p_to_noul_answer(float(probs[1] if len(probs) > 1 else probs[0]))
        elif q["type"] == "score":
            answers[qid] = _probs_to_score_answer(q, probs)
        else:
            raise ValueError(q["type"])
        n_input_tokens += len(str(q).split())
    dt_ms = (time.perf_counter() - t0) * 1000
    return {
        "model": model_name,
        "answers": answers,
        "usage": {"input_tokens": n_input_tokens, "output_tokens": 0},
        "latency_ms": dt_ms,
    }
