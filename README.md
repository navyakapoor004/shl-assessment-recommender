# SHL Assessment Recommender (Conversational Agent)

Ek conversational recommender jo vague hiring intent ("Java developer hire kar raha hoon")
ko dialogue ke through SHL catalog se grounded 1–10 assessment shortlist mein convert karta hai.
Anti-hallucination by design — writer LLM sirf retrieved real catalog items dekhta hai, koi bhi
URL jo catalog mein nahi hai wo response mein kabhi nahi jaa sakta.

## Architecture

```
User + full history
   ↓
CALL A: Router LLM (Groq, JSON) → { route, constraints }
   ↓
route == CLARIFY → question poocho, empty recs
route == REFUSE  → canned safe reply, empty recs
route == COMPARE → catalog se 2 named items lookup, grounded diff
route == RECOMMEND/REFINE:
      ↓
   Hard metadata filter (test_type, level, duration)
      ↓
   Hybrid retrieve (FAISS semantic + keyword boost) → top 10
      ↓
   Grounding gate (sirf catalog URLs allowed)
      ↓
CALL B: Writer LLM (Groq) → grounded reply, sirf 10 real items dekhta hai
      ↓
Pydantic schema validate → strict JSON return
```

Stateless service — har `/chat` call mein poori conversation history client bhejta hai,
server kuch save nahi karta.

## Stack

- **FastAPI + Pydantic** — API + non-negotiable strict JSON schema
- **Groq** — router LLM (Call A) + writer LLM (Call B), dono `llama-3.1-8b-instant` (fast + cheap)
- **sentence-transformers + FAISS** — offline embeddings + semantic retrieval
- **BeautifulSoup + requests** — catalog scraper

Django nahi liya kyunki ye heavy hai — ORM/admin/templates ki yahan zaroorat nahi. FastAPI
lightweight + async hai, aur Pydantic se strict schema built-in milta hai.

## Project structure

```
shl-recommender/
├── app/
│   ├── main.py                  # FastAPI app, /chat + /health
│   ├── schemas.py                # ALL Pydantic models (request/response/router/catalog)
│   └── services/
│       ├── llm.py                 # Groq calls: router + writer
│       ├── retrieval.py           # FAISS + hybrid search + hard filters
│       └── orchestrator.py        # Ties everything together + grounding gate
├── data/
│   └── catalog.json               # Ground truth (sample data included — see below)
├── scripts/
│   ├── scrape_catalog.py          # Real SHL scraper (run locally, not in sandbox)
│   └── build_index.py             # One-time offline FAISS index pre-build
├── tests/
│   └── test_api.py                # Pytest + mocked Groq calls
├── requirements.txt
├── Dockerfile                     # For Render / HF Spaces
├── .env.example
└── README.md
```

## Setup

```bash
cd shl-recommender
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# .env mein apna GROQ_API_KEY daalo (free key: https://console.groq.com/keys)
export GROQ_API_KEY=sk-...      # ya .env se python-dotenv load karo
```

### Real catalog scrape karna (optional — sample catalog.json already included hai)

```bash
python scripts/scrape_catalog.py
```

Ye `data/catalog.json` ko real SHL product catalog se overwrite kar dega. Sandbox mein SHL
ki site allowed-domain list mein nahi thi isliye maine `data/catalog.json` mein 15 realistic
SHL-style placeholder entries daali hain taaki tum turant test kar sako. Scraper selectors
defensive hain (fallbacks ke saath) but SHL ka markup change ho sakta hai — agar kuch match
na ho to live page HTML dekh kar `# SELECTOR` comments wali lines update karna.

### Run karna

```bash
uvicorn app.main:app --reload --port 8000
```

Test:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need to hire a Java developer, mid-level, test should be under 30 minutes"}]}'

curl http://localhost:8000/health
```

## Response schema (fixed contract)

```json
{
  "reply": "string",
  "recommendations": [
    {"name": "string", "url": "string", "test_type": "coding"}
  ],
  "end_of_conversation": false
}
```

`recommendations` empty rehta hai jab route CLARIFY ya REFUSE ho.

## Testing

```bash
pytest -q
```

LLM calls mocked hain isliye tests offline/free chalte hain — retrieval + grounding gate +
schema validation end-to-end test hoti hai.

## Deploy (Render / Hugging Face Spaces)

Dockerfile already included hai (port `$PORT` ya `7860` dono handle karta hai).

**Render:** New Web Service → connect repo → Environment: Docker → env var `GROQ_API_KEY` set
karo → deploy.

**HF Spaces:** New Space → SDK: Docker → repo push karo → Space settings mein `GROQ_API_KEY`
secret add karo.

Deploy hone ke baad `GET /health` check karna — `{"status": "ok"}` aana chahiye.

## Anti-hallucination guarantee

Writer LLM (Call B) ko sirf `retrieved_items` (jo catalog se hard-filtered + FAISS-retrieved
hain) dikhaye jaate hain — kabhi bhi poora catalog ya open-ended generation nahi. Final
`recommendations` list bhi LLM ke text output se nahi, seedha `retrieved_items` se construct
hoti hai (`app/services/orchestrator.py::_handle_recommend`), matlab LLM chaahe bhi to
non-existent URL response mein daal hi nahi sakta.

## Notes / next steps

- `data/catalog.json` mein abhi 15 sample entries hain — real scrape ke baad ye number 100+
  ho sakta hai depending on SHL's actual catalog size.
- `GROQ_MODEL` env var se model switch kar sakti ho (e.g. `llama-3.3-70b-versatile` better
  quality ke liye, thoda slower).
- Multi-turn REFINE state client-side history se hi aata hai (stateless server design) — agar
  future mein session-based memory chahiye to Redis/Postgres add karna hoga.
