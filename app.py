"""
app.py - GenAI Prompt Gateway (Streamlit)

Checks a staff prompt BEFORE it is sent to the company's internal GenAI assistant:
  Pipeline 1 (fine-tuned): text-classification  -> prompt injection / jailbreak?
  Pipeline 2 (pretrained): token-classification -> personal data (PII) in the prompt?

Decision:  BLOCK  if an injection is detected
           REVIEW if personal data is found
           ALLOW  otherwise
"""

import pandas as pd
import streamlit as st
from transformers import pipeline

# ---------------------------------------------------------------------------
# Settings - update the two model names after running the experiment scripts
# ---------------------------------------------------------------------------
COMPANY_NAME = "Example Bank (Hong Kong)"                       # replace with the chosen company
INJECTION_MODEL = "your-hf-username/prompt-gateway-injection"   # our fine-tuned model on HF Hub
PII_MODEL = "Isotonic/distilbert_finetuned_ai4privacy_v2"       # PII model chosen in predict_base.py

BLOCK_THRESHOLD = 0.5     # injection probability at or above this value -> BLOCK
MIN_ENTITY_SCORE = 0.5    # ignore PII entities found with lower confidence
MAX_CHARS = 1500          # longest prompt accepted (keeps inputs within the model limit)
MAX_BATCH_ROWS = 100      # longest CSV accepted in the batch test tab

INJECTION_WORDS = ("INJECTION", "JAILBREAK", "MALICIOUS", "UNSAFE")
NON_PII_LABELS = {
    "ORG", "MISC", "COMPANYNAME", "JOBAREA", "JOBTITLE", "JOBTYPE",
    "CURRENCY", "CURRENCYCODE", "CURRENCYNAME", "CURRENCYSYMBOL",
    "AMOUNT", "DATE", "TIME", "ORDINALDIRECTION",
}


# ---------------------------------------------------------------------------
# Model loading (cached so models load only once per app session)
# ---------------------------------------------------------------------------
def get_hf_token():
    """Read an optional Hugging Face token from Streamlit secrets.
    Only needed if the fine-tuned model repo is set to private."""
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading models, please wait ...")
def load_models():
    token = get_hf_token()
    injection_clf = pipeline("text-classification", model=INJECTION_MODEL, token=token)
    pii_ner = pipeline("token-classification", model=PII_MODEL,
                       aggregation_strategy="simple", token=token)
    return injection_clf, pii_ner


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def is_injection_label(label):
    label = str(label).upper()
    return label in ("LABEL_1", "1") or any(word in label for word in INJECTION_WORDS)


def clean_label(label):
    label = str(label).upper()
    return label[2:] if label[:2] in ("B-", "I-") else label


def injection_probability(clf, text):
    """Probability (0-1) that the prompt is an injection."""
    scores = clf(text, top_k=None, truncation=True, max_length=512)
    if scores and isinstance(scores[0], list):   # some versions return a nested list
        scores = scores[0]
    return sum(s["score"] for s in scores if is_injection_label(s["label"]))


def find_pii(ner, text):
    """List of personal data entities found in the prompt."""
    entities = ner(text)
    return [e for e in entities
            if e["score"] >= MIN_ENTITY_SCORE and clean_label(e["entity_group"]) not in NON_PII_LABELS]


def mask_text(text, entities):
    """Replace each PII entity with its type, e.g. [EMAIL]."""
    for e in sorted(entities, key=lambda x: x["start"], reverse=True):
        text = text[:e["start"]] + f"[{clean_label(e['entity_group'])}]" + text[e["end"]:]
    return text


def check_prompt(text, injection_clf, pii_ner):
    """Run both pipelines and return the gateway decision."""
    prob = injection_probability(injection_clf, text)
    entities = find_pii(pii_ner, text)
    if prob >= BLOCK_THRESHOLD:
        decision = "BLOCK"
    elif entities:
        decision = "REVIEW"
    else:
        decision = "ALLOW"
    return {"decision": decision, "injection_prob": prob, "entities": entities}


def validate_prompt(text):
    """Return an error message, or None if the prompt is acceptable."""
    if not text or not text.strip():
        return "Please enter a prompt."
    if len(text) > MAX_CHARS:
        return f"Prompt is too long ({len(text)} characters). The limit is {MAX_CHARS}."
    return None


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="GenAI Prompt Gateway", page_icon="🛡️", layout="wide")
st.title("GenAI Prompt Gateway")
st.caption(f"Pre-check for staff prompts to the internal GenAI assistant at {COMPANY_NAME}")

with st.sidebar:
    st.subheader("Models in use")
    st.write(f"**Injection check (fine-tuned):** {INJECTION_MODEL}")
    st.write(f"**Personal data check:** {PII_MODEL}")
    st.write(f"Block threshold: {BLOCK_THRESHOLD:.0%}")
    st.info("BLOCK = possible injection. REVIEW = personal data found. ALLOW = no issue found.")

try:
    injection_clf, pii_ner = load_models()
except Exception as err:
    st.error(f"The models could not be loaded. Please check the model names in app.py. Details: {err}")
    st.stop()

tab_single, tab_batch = st.tabs(["Check a prompt", "Batch test (CSV)"])

# ----- Tab 1: one prompt -----
with tab_single:
    prompt = st.text_area("Paste the prompt that will be sent to the GenAI assistant:", height=150)
    if st.button("Check prompt", type="primary"):
        error = validate_prompt(prompt)
        if error:
            st.warning(error)
        else:
            try:
                result = check_prompt(prompt, injection_clf, pii_ner)
            except Exception as err:
                st.error(f"Something went wrong while checking the prompt: {err}")
                st.stop()

            prob, entities = result["injection_prob"], result["entities"]
            if result["decision"] == "BLOCK":
                st.error(f"BLOCK: this prompt looks like an attempt to override the assistant's "
                         f"instructions ({prob:.0%} confidence). It should not be sent.")
            elif result["decision"] == "REVIEW":
                st.warning(f"REVIEW: no injection was detected, but the prompt contains "
                           f"{len(entities)} item(s) of personal data. Remove or mask them before sending.")
            else:
                st.success("ALLOW: no injection and no personal data were found. The prompt can be sent.")

            summary = pd.DataFrame([
                {"Check": "Prompt injection", "Result": "Detected" if prob >= BLOCK_THRESHOLD else "Not detected",
                 "Confidence": f"{max(prob, 1 - prob):.0%}"},
                {"Check": "Personal data", "Result": f"{len(entities)} item(s) found" if entities else "None found",
                 "Confidence": f"{max(e['score'] for e in entities):.0%}" if entities else "-"},
            ])
            st.table(summary)

            if entities:
                st.subheader("Personal data found")
                st.table(pd.DataFrame([{"Type": clean_label(e["entity_group"]), "Text": e["word"],
                                        "Confidence": f"{e['score']:.0%}"} for e in entities]))
                st.subheader("Masked version of the prompt")
                st.code(mask_text(prompt, entities), language=None)

# ----- Tab 2: batch test for live-app evaluation -----
with tab_batch:
    st.write("Upload a CSV with a **text** column. Optional columns **expected_injection** and "
             "**expected_pii** (0 or 1) are used to calculate accuracy.")
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as err:
            st.error(f"Could not read the CSV file: {err}")
            st.stop()

        if "text" not in df.columns:
            st.error("The CSV must have a column named 'text'.")
        elif len(df) > MAX_BATCH_ROWS:
            st.error(f"Please upload at most {MAX_BATCH_ROWS} rows.")
        else:
            rows, progress = [], st.progress(0.0, text="Checking prompts ...")
            for i, text in enumerate(df["text"].astype(str)):
                if validate_prompt(text):
                    rows.append({"decision": "SKIPPED", "injection_pred": None, "pii_pred": None,
                                 "injection_prob": None})
                else:
                    r = check_prompt(text, injection_clf, pii_ner)
                    rows.append({"decision": r["decision"],
                                 "injection_pred": int(r["injection_prob"] >= BLOCK_THRESHOLD),
                                 "pii_pred": int(bool(r["entities"])),
                                 "injection_prob": round(r["injection_prob"], 3)})
                progress.progress((i + 1) / len(df), text=f"Checked {i + 1} of {len(df)}")

            out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

            col1, col2 = st.columns(2)
            for col, check in ((col1, "injection"), (col2, "pii")):
                if f"expected_{check}" in out.columns:
                    valid = out.dropna(subset=[f"{check}_pred"])
                    correct = (valid[f"expected_{check}"] == valid[f"{check}_pred"])
                    col.metric(f"{'Injection' if check == 'injection' else 'Personal data'} accuracy",
                               f"{correct.mean():.0%}", f"{correct.sum()} of {len(valid)} correct")
                    out[f"{check}_correct"] = out[f"expected_{check}"] == out[f"{check}_pred"]

            st.dataframe(out, use_container_width=True)
            st.download_button("Download results (CSV)", out.to_csv(index=False),
                               file_name="live_app_results.csv", mime="text/csv")
