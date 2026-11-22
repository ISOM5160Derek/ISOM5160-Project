# GenAI Prompt Gateway

Streamlit app that checks a staff prompt before it reaches an internal GenAI assistant.

- Pipeline 1 (fine-tuned): prompt injection / jailbreak detection (text-classification)
- Pipeline 2 (pretrained): personal data detection (token-classification)

Run locally:

    pip install -r requirements.txt
    streamlit run app.py

`test_cases/live_test_cases.csv` can be uploaded in the "Batch test" tab to measure live-app accuracy.
