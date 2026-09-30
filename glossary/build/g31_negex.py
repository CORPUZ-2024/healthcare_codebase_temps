"""
G31 — Rule-based clinical concept extraction with negation, history and family context (NegEx-style)
=====================================================================================================
Copy this whole block into a file (e.g. negex.py) and run:  python negex.py
Requires: pandas   (pip install pandas)   Production tooling: medspaCy (ConText), cTAKES, or a validated NLP service.
"""
import re

import pandas as pd

PRE_NEG = ["no", "denies", "denied", "without", "negative for", "no evidence of", "no signs of", "not", "free of"]
POST_NEG = ["was ruled out", "is ruled out", "unlikely", "not present"]
HISTORICAL = ["history of", "h/o", "previous", "prior"]
FAMILY = ["mother", "father", "sister", "brother", "family history of", "fh of"]
TERMINATORS = ["but", "however", "although", "except", "aside from", ";"]


def extract(notes: pd.DataFrame, concepts: dict[str, list[str]], window: int = 6) -> pd.DataFrame:
    """Find concept mentions in free-text notes and label each as negated / historical / family / affirmed.

    Healthcare context
    ------------------
    Care-management and home-care notes hold information claims don't: falls, caregiver strain,
    wandering, food insecurity. Keyword search alone is wrong half the time: "denies falls", "no
    caregiver strain", "mother had dementia". NegEx-style rules check a few words before (and after)
    each mention for negation, history or family triggers, stopping at scope terminators ("but").

    Parameters
    ----------
    notes : note_id, text; concepts : {concept name: [synonyms]}

    Returns note_id, sentence_no, concept, matched_text, negated_flag, historical_flag, family_flag, affirmed_flag.

    Steps
    -----
    1. Lower-case; split into sentences on . ! ? and newlines.
    2. Find each synonym (word boundaries). Scope = up to ``window`` tokens before the mention,
       cut at the last terminator; plus the rest of the sentence for post-negation triggers.
    3. Flags from trigger lists; affirmed = none of negated / historical / family.

    Common mistakes
    ---------------
    - Substring matching ("fall" inside "fallopian"); use word boundaries.
    - Unlimited scope ("no fever, but reports falls" - "no" must not negate "falls").
    - Evaluating on the notes you wrote the rules from; validate precision/recall on a held-out, annotated set.
    """
    rows = []
    for note in notes.itertuples(index=False):
        for s_no, sent in enumerate(re.split(r"[.!?\n]+", note.text.lower())):                        # step 1
            for concept, syns in concepts.items():
                for syn in syns:
                    for m in re.finditer(rf"\b{re.escape(syn)}\b", sent):                                   # step 2
                        before = sent[: m.start()]
                        for t in TERMINATORS:
                            cut = before.rfind(f" {t} ") if t != ";" else before.rfind(";")
                            if cut != -1:
                                before = before[cut + len(t) + 1:]
                        pre = " ".join(before.split()[-window:])
                        after = sent[m.end():]
                        has = lambda triggers, text: any(re.search(rf"\b{re.escape(x)}\b", text) for x in triggers)  # noqa: E731
                        neg = has(PRE_NEG, pre) or has(POST_NEG, after)                                    # step 3
                        hist, fam = has(HISTORICAL, pre), has(FAMILY, pre)
                        rows.append({"note_id": note.note_id, "sentence_no": s_no, "concept": concept, "matched_text": syn,
                                     "negated_flag": int(neg), "historical_flag": int(hist), "family_flag": int(fam),
                                     "affirmed_flag": int(not (neg or hist or fam))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Self-test: run `python negex.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    concepts = {"fall": ["fall", "falls", "fell"], "caregiver_strain": ["caregiver strain", "caregiver burnout"],
                "dementia": ["dementia"], "fever": ["fever"]}
    notes = pd.DataFrame({"note_id": [1, 2, 3, 4, 5, 6], "text": [
        "Patient fell in the bathroom last night.",
        "Denies falls. Daughter reports caregiver burnout.",
        "Mother had dementia. Patient is alert and oriented.",
        "No fever, but reports two falls this week.",
        "History of fall with hip fracture in 2019. Pneumonia was ruled out; fever unlikely.",
        "Fallopian tube imaging normal.",
    ]})
    r = extract(notes, concepts)
    print(r)
    get = lambda nid, c: r[(r.note_id == nid) & (r.concept == c)].iloc[0]  # noqa: E731
    checks = {
        "note 1: 'fell' affirmed": get(1, "fall").affirmed_flag == 1,
        "note 2: 'denies falls' negated": get(2, "fall").negated_flag == 1,
        "note 2: caregiver burnout affirmed": get(2, "caregiver_strain").affirmed_flag == 1,
        "note 3: mother's dementia is family history, not the patient's": get(3, "dementia").family_flag == 1,
        "note 4: 'no' negates fever but NOT falls after 'but'": get(4, "fever").negated_flag == 1 and get(4, "fall").affirmed_flag == 1,
        "note 5: 'history of fall' historical; 'fever unlikely' negated (post-trigger)":
            get(5, "fall").historical_flag == 1 and get(5, "fever").negated_flag == 1,
        "word boundaries: 'fallopian' is not a fall": (r.note_id == 6).sum() == 0,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
