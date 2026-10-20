"""Response scoring. Raw text is always kept; normalization is a separate step."""
import re

PROMPT = ("Read the race bib number of the central foreground runner. "
          "Return only its digits, or UNKNOWN if unreadable.")


def extract_bib_digits(raw_text):
    """Legacy rule: first standalone digit group, else first digit group anywhere."""
    standalone = re.findall(r"\b\d+\b", raw_text or "")
    if standalone:
        return standalone[0]
    anywhere = re.findall(r"\d+", raw_text or "")
    return anywhere[0] if anywhere else ""


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_response(raw, label):
    """Score one successful generation. Categories overlap; see the flags.

    cer is true Levenshtein distance of the normalized prediction over len(label).
    """
    raw = "" if raw is None else str(raw)
    stripped = raw.strip()
    normalized = extract_bib_digits(raw)
    return {
        "raw": raw,
        "normalized": normalized,
        "strict_exact": stripped == label,
        "normalized_exact": normalized == label,
        "digits_only": re.fullmatch(r"\d+", stripped) is not None,
        "abstained": stripped.rstrip(".").upper() == "UNKNOWN",
        "empty": stripped == "",
        "no_digits": normalized == "",
        "multiple_digit_groups": len(re.findall(r"\d+", raw)) > 1,
        "cer": levenshtein(normalized, label) / len(label) if label else None,
    }


def summarize(trials, planned):
    """trials: list of {status: ok|failed, score: dict|None}. planned: planned trial count.

    Failed and unattempted trials stay in the denominator.
    """
    ok = [t for t in trials if t.get("status") == "ok" and t.get("score")]
    failed = sum(1 for t in trials if t.get("status") == "failed")
    s = [t["score"] for t in ok]
    count = lambda k: sum(1 for x in s if x[k])
    return {
        "planned": planned,
        "attempted": len(trials),
        "unattempted": planned - len(trials),
        "succeeded": len(ok),
        "failed": failed,
        "strict_exact": count("strict_exact"),
        "normalized_exact": count("normalized_exact"),
        "digits_only": count("digits_only"),
        "abstained": count("abstained"),
        "empty": count("empty"),
        "no_digits": count("no_digits"),
        "multiple_digit_groups": count("multiple_digit_groups"),
        "mean_cer_successful": (sum(x["cer"] for x in s) / len(s)) if s else None,
        "cer_denominator": "successful trials only; failures reported separately",
    }
