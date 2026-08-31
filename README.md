# MediGuide AI

AI-powered preliminary medical symptom assessment and patient guidance
assistant, built with **LangChain** + **Streamlit**.

> ⚠️ **Educational prototype only.** This is not a medical device and must
> never be used for real diagnosis or treatment. Always consult a licensed
> healthcare professional, and seek emergency care immediately in an
> emergency.

## What it does

1. Collects patient age, gender, symptoms, duration, severity, existing
   conditions, medications, and notes through a Streamlit form.
2. Sends the data through a LangChain `LLMChain` (built from a
   `ChatPromptTemplate`) to an OpenAI chat model, asking for a structured
   JSON assessment.
3. Safely parses the JSON (never crashes on malformed output).
4. Streams a friendly, plain-language narrative live into the page with
   `st.write_stream`.
5. Renders a results dashboard: urgency level, possible conditions,
   recommended next steps, questions for your doctor, and warning signs.
6. Demonstrates both `InMemoryCache` and `SQLiteCache` so repeated,
   identical requests are answered instantly.

## Project structure

```
medical_ai_assistant/
├── app.py                  # Streamlit UI (run this)
├── requirements.txt
├── .env.example
├── README.md
└── src/
    ├── __init__.py
    ├── config.py            # settings + form dropdown options
    ├── prompts.py            # PromptTemplate + ChatPromptTemplate + JSON schema
    ├── chains.py              # ChatOpenAI, LLMChain, streaming, message demo
    ├── cache_manager.py       # InMemoryCache / SQLiteCache switch
    └── utils.py               # safe JSON parsing + helpers
```

## Getting started

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

### API key

You have two options:

- **Bring-your-own-key (default):** on first launch you'll see a
  "Connect your OpenAI key" screen. Paste a key from
  [platform.openai.com](https://platform.openai.com). It's kept only in
  `st.session_state` for that session — never written to disk.
- **.env for local/dev use:** copy `.env.example` to `.env` and set
  `OPENAI_API_KEY=sk-...`. If this is present, the gate screen is skipped
  automatically. Never commit your real `.env` file.

## Caching: InMemoryCache vs SQLiteCache

| | InMemoryCache | SQLiteCache |
|---|---|---|
| Stored in | RAM (a Python dict) | A file on disk (`.langchain_cache.db`) |
| Speed | Fastest | Fast, marginally slower (disk I/O) |
| Survives restart? | No — cleared on restart | Yes — persists across runs |
| Best for | A single working session | Reusing cached answers across sessions/days |

Both are registered once via `langchain.globals.set_llm_cache(...)`.
Once registered, LangChain automatically checks the cache **before** every
model call and stores new responses **after**. Submitting the exact same
form twice will be visibly faster the second time, and the response-time
caption in the UI shows this directly. Switch the mode from the sidebar's
"Cache mode" dropdown at any time.

## Testing scenarios

| # | Input | Expected behaviour |
|---|---|---|
| 1 | Age 25, runny nose + sore throat, 1-3 days, severity 2 | Urgency LOW; calm monitoring advice |
| 2 | Age 40, fever + cough, 4-7 days, severity 6 | Urgency MEDIUM/HIGH; advises seeing a professional |
| 3 | Severe chest pain + shortness of breath | Urgency HIGH/EMERGENCY; urges immediate help |
| 4 | Submit the same form twice (cache on) | Second run is faster; identical result |
| 5 | Empty symptoms | App warns the user and does not call the API |
| 6 | Language = Urdu | Guidance text returns in Urdu |

## Notes

- The `possible_conditions` list is for education only and is never
  presented as a diagnosis — every screen carries the safety disclaimer.
- If the model ever returns malformed JSON, `src/utils.safe_json_parse`
  catches it and the UI shows a friendly error plus the raw output for
  debugging, instead of crashing.
