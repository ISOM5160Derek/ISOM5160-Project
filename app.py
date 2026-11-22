"""
app.py - GenAI Prompt Gateway (Streamlit)

Tab 1 (staff):          chat with the internal assistant; every message passes the gateway first
Tab 2 (security team):  check one prompt in detail
Tab 3 (governance team): batch test with a CSV to measure accuracy

The gateway checks each prompt BEFORE it is sent to the company's internal GenAI assistant:
  Pipeline 1 (fine-tuned): text-classification  -> prompt injection / jailbreak?
  Pipeline 2 (pretrained): token-classification -> personal data (PII) in the prompt?

Decision:  BLOCK  if an injection is detected
           REVIEW if personal data is found
           ALLOW  otherwise
"""

import pandas as pd
import requests
import streamlit as st
from transformers import pipeline

# ---------------------------------------------------------------------------
# Settings - update the two model names after running the experiment scripts
# ---------------------------------------------------------------------------
COMPANY_NAME = "Example Bank (Hong Kong)"                       # replace with the chosen company
INJECTION_MODEL = "Claudeface/prompt-gateway-injection"   # our fine-tuned model on HF Hub
PII_MODEL = "Isotonic/distilbert_finetuned_ai4privacy_v2"       # PII model chosen in predict_base.py

BLOCK_THRESHOLD = 0.5     # injection probability at or above this value -> BLOCK
MIN_ENTITY_SCORE = 0.5    # ignore PII entities found with lower confidence
MAX_CHARS = 1500          # longest prompt accepted (keeps inputs within the model limit)
MAX_BATCH_ROWS = 100      # longest CSV accepted in the batch test tab

# Chat tab settings
MAX_CHAT_TURNS = 20       # messages per session sent to a live chat model (protects API credit)
SIMULATED_REPLY = ("(Demo) Your message passed the gateway and was forwarded to the internal "
                   "assistant. In production, the assistant's answer would appear here.")

INJECTION_WORDS = ("INJECTION", "JAILBREAK", "MALICIOUS", "UNSAFE")
# Entity types treated as personal data (same allow-list as the experiment scripts)
PII_LABELS = {
    "PER", "NAME", "FULLNAME", "FIRSTNAME", "MIDDLENAME", "LASTNAME",
    "EMAIL", "PHONENUMBER", "PHONE_NUMBER", "PHONEIMEI", "USERNAME", "PASSWORD",
    "ACCOUNTNUMBER", "IBAN", "BIC", "CREDITCARDNUMBER",
    "CREDITCARDCVV", "MASKEDNUMBER", "PIN", "SSN", "DOB",
    "STREET", "STREETADDRESS", "BUILDINGNUMBER", "SECONDARYADDRESS", "ZIPCODE",
    "IP", "IPV4", "IPV6", "MAC", "NEARBYGPSCOORDINATE",
    "BITCOINADDRESS", "ETHEREUMADDRESS", "LITECOINADDRESS", "VEHICLEVIN", "VEHICLEVRM",
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
            if e["score"] >= MIN_ENTITY_SCORE and clean_label(e["entity_group"]) in PII_LABELS]


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


def get_assistant_settings():
    """Optional live chat model, read from Streamlit secrets (OpenAI-compatible API).
    If the secrets are missing, the chat tab uses a simulated reply."""
    try:
        return {"api_key": st.secrets["ASSISTANT_API_KEY"],
                "base_url": st.secrets["ASSISTANT_BASE_URL"].rstrip("/"),
                "model": st.secrets["ASSISTANT_MODEL"]}
    except Exception:
        return None


def assistant_reply(message, settings):
    """Answer from the downstream assistant. Only called for prompts the gateway ALLOWED."""
    if settings is None:
        return SIMULATED_REPLY
    try:
        response = requests.post(
            f"{settings['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {settings['api_key']}"},
            json={"model": settings["model"], "max_tokens": 300,
                  "messages": [{"role": "system",
                                "content": f"You are the internal assistant of {COMPANY_NAME}. "
                                           "Answer briefly and professionally."},
                               {"role": "user", "content": message}]},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return SIMULATED_REPLY + "\n\n(The live assistant could not be reached, so a demo reply is shown.)"


def gateway_chat_reply(message, injection_clf, pii_ner, settings):
    """Return (text to keep in the chat history, assistant reply) for one staff message."""
    result = check_prompt(message, injection_clf, pii_ner)
    if result["decision"] == "BLOCK":
        return message, ("🚫 **Blocked by the gateway.** This message looks like an attempt to "
                         f"override the assistant's rules ({result['injection_prob']:.0%} confidence). "
                         "It was not sent. Contact the AI governance team if you think this is wrong.")
    if result["decision"] == "REVIEW":
        masked = mask_text(message, result["entities"])
        types = ", ".join(sorted({clean_label(e["entity_group"]) for e in result["entities"]}))
        # Only the masked text is kept in the chat history, so personal data is not stored
        return masked, ("⚠️ **Not sent: personal data found** (" + types + "). "
                        "Please remove it and send again. Masked version:\n\n> " + masked)
    if settings is not None and st.session_state.get("turns", 0) >= MAX_CHAT_TURNS:
        return message, "✅ Passed the gateway, but the demo message limit for this session was reached."
    st.session_state["turns"] = st.session_state.get("turns", 0) + 1
    return message, "✅ *Passed the gateway.*\n\n" + assistant_reply(message, settings)


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

tab_chat, tab_single, tab_batch = st.tabs(["Staff assistant (chat)", "Check a prompt", "Batch test (CSV)"])

# ----- Tab 1: staff chat, every message goes through the gateway first -----
with tab_chat:
    settings = get_assistant_settings()
    st.caption("Every message is checked by the gateway before it reaches the assistant. "
               "Assistant mode: " + ("live chat model" if settings else "simulated reply (demo)"))
    if "chat" not in st.session_state:
        st.session_state["chat"] = []        # list of {"role": ..., "content": ...}

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Type your message to the assistant")
    if user_msg:
        error = validate_prompt(user_msg)
        if error:
            st.warning(error)
        else:
            try:
                shown, reply = gateway_chat_reply(user_msg, injection_clf, pii_ner, settings)
            except Exception as err:
                shown, reply = user_msg, f"Something went wrong while checking the message: {err}"
            st.session_state["chat"] += [{"role": "user", "content": shown},
                                         {"role": "assistant", "content": reply}]
            st.rerun()

    if st.session_state["chat"] and st.button("Clear chat"):
        st.session_state["chat"] = []
        st.rerun()

# ----- Tab 2: one prompt in detail (security analyst) -----
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

# ----- Tab 3: batch test for live-app evaluation (governance team) -----
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
