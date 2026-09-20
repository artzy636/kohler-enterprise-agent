KOHLER Unified Enterprise AI Agent

An enterprise AI agent for HR, Finance, Customer Support, and Legal/Compliance questions. Built for the KOHLER-MITWPU AI Research Centre case study (Track 3).

The main thing I wanted to get right here: instead of an assistant that always sounds confident, this one actually checks its own answer against the source documents before replying. If it's not sure, it asks a follow-up question or says it's escalating to the right team, instead of making something up.

Try it live

kohler-enterprise-agent.streamlit.app

Just open the link, no setup needed. A couple of things you might notice:

If nobody's used it in a while, the first load can take 30-60 seconds — the free hosting tier puts idle apps to sleep and it has to wake back up. Not a bug, just how the free tier works.
Each answer takes about 10-15 seconds to come back. That's intentional — I'm rate-limiting calls myself so we don't blow through Gemini's free-tier quota (more on that below).

Everything after this is for running it locally / reading the code, not needed if you just want to try the app.

How it's put together

```
User query
│
▼
Domain Router — figures out if this is HR / Finance / Support / Legal (or none of these)
│
▼
Retrieval — pulls the most relevant chunks from that domain's knowledge base (FAISS)
│
▼
Draft Agent — writes an answer using only what was retrieved
│
▼
Verification Agent — double-checks the draft against the source text, decides:
answer it / ask for clarification / escalate to a human
│
▼
Response Routing — sends back whichever of those three it is, with citations if it's a real answer
│
▼
Format Layer — same answer, but as plain text / JSON / Excel / XML / an email draft
```

It also remembers the last few messages in the conversation, so a follow-up like "what about parental leave?" after asking about vacation days actually works instead of being treated as a brand new, context-less question.

What it's built with
LLM: Gemini (google-genai SDK) — currently running everything on Flash-Lite
Embeddings: sentence-transformers (all-MiniLM-L6-v2), done locally, no API call needed for this part
Vector search: FAISS, a separate index per domain
Frontend: Streamlit, with some custom theming

One thing worth explaining rather than glossing over: I originally split this across two model tiers — a stronger model for writing answers, Flash-Lite for the cheaper routing/verification steps. That fell apart fast once I actually started testing, because the stronger tier's free daily quota turned out to be 20 requests a day, which I burned through almost immediately. Flash-Lite's quota was a lot more forgiving, so I moved everything onto it. Not the plan I started with, but a real decision made from actually hitting the constraint rather than guessing at it upfront — and none of the architecture would need to change if this ran on a paid plan with a stronger model.

The knowledge base

Four domains, 12 documents each: HR, Finance, Customer Support, and Legal/Compliance. The Customer Support docs are based on real KOHLER products — Anthem EvoCycle, Numi 2.0, the Leap collection, Verdera Voice — paraphrased in my own words rather than copied from anywhere. HR/Finance/Legal are made up but written to be realistic, and I deliberately wrote in some overlaps and gaps: a few questions that need two documents combined to answer properly, some genuinely ambiguous ones, and a few that aren't covered by anything in the knowledge base at all — mainly so I'd have real cases to test the escalate/clarify logic against, not just easy wins.

Quick note on scope: the brief lists five domains and keeps privacy separate from legal/compliance. I merged them, because in practice the privacy content (data retention, breach reporting, that kind of thing) overlaps so much with legal/compliance that splitting them would've meant writing mostly the same material twice rather than adding anything new.

Running it yourself
Clone this repo
Install the dependencies:
```
pip install -r requirements.txt
```
Grab a free Gemini API key from aistudio.google.com — no card needed
Make a .env file in the project root:
```
GEMINI_API_KEY=your-key-here
```
Build the search indexes (only needs doing once, or again if you edit anything in data/):
```
python src/build_index.py
```
Run it:
```
streamlit run app.py
```
Testing it

eval/test_set.json has 20 test questions I wrote by hand — some straightforward, some that need two documents to answer, some deliberately ambiguous, and a few that the knowledge base just doesn't cover at all.

Run it with:
```
python eval/run_eval.py
```

Current result: 15 out of 20 (75%). Every single miss comes down to the same thing: the system disagreeing with my own labeling on whether an ambiguous question should get answered directly, get a clarifying question, or get escalated — all three are defensible responses to those particular questions. What I actually care about more: across every run I did while building this, it never once made something up or stated a fact that wasn't true. When it doesn't know something, it says so, consistently.

Things that aren't perfect, and why
It's slow (~10-15s per answer). That's the free-tier rate limiting mentioned above, not a design flaw — on a paid plan this delay just goes away.
The answer/clarify/escalate call isn't 100% consistent on ambiguous questions. Ask the same borderline question twice and you might get two different (but both reasonable) responses. This is just how LLMs behave on genuinely fuzzy cases — it's not something I could fully eliminate, only manage.
Where everything lives

```
data/ The knowledge base itself (4 domains, 12 docs each)
indexes/ FAISS indexes, generated by build_index.py
src/
build_index.py Builds the embeddings and indexes from data/
retrieve.py Retrieval logic
agents.py The router, draft, and verification agents
pipeline.py Wires everything together into one answer_query() call
formatters.py Turns an answer into JSON/Excel/XML/email
eval/
test_set.json The 20 test questions
run_eval.py Runs them and reports pass/fail
app.py The Streamlit app itself
.streamlit/config.toml Custom theme
```
## Submission materials

- Prompts documentation: [docs/KOHLER_Track3_Prompts_Documentation.pdf](docs/KOHLER_Track3_Prompts_Documentation.pdf)
- Presentation deck: [docs/KOHLER_Track3_Deck.pdf](docs/KOHLER_Track3_Deck.pdf)
- Demo video: [docs/KOHLER_Track3_Demo_Video(fl).mp4](docs/KOHLER_Track3_Demo_Video(fl).mp4)

## What's in scope, and what to try

The knowledge base covers four domains — **HR, Finance, Product Support, and Legal** — across 48 hand-authored documents. Anything outside that scope is designed to escalate to a human rather than guess.

**Answered directly** (grounded in the knowledge base):
- "How many days of annual leave do full-time employees accrue per year?"
- "How long is the warranty on the Numi 2.0 smart toilet?"
- "I'm a manager and want to buy a $12,000 piece of equipment. What approval process applies?"

**Asks a clarifying question** (needs one detail about you specifically):
- "What's my per diem for this trip?"
- "Is my product still under warranty?"
- "How much can I approve for a purchase without needing additional sign-off?"

**Escalates to a human** (genuinely out of scope — the system won't guess):
- "Does Kohler offer an employee referral bonus, and how much is it?"
- "What's the vesting schedule for employee stock options?"

**Multi-turn memory** — ask a question, then a pronoun-based follow-up (e.g. "How long is the Numi 2.0's warranty?" → "What about the Leap toilet instead?") to see it resolve the follow-up using conversation history.

The full 20-case test set with all expected outcomes is in `eval/test_set.json`, and the detailed reasoning behind each routing decision is in the prompts documentation.

## Knowledge base contents

The assistant's knowledge base is 48 short, hand-authored markdown documents across four domains — nothing outside these topics is answerable; anything else correctly escalates.

**HR** (12 documents): leave policy, parental leave, benefits overview, onboarding, resignation process, performance review cycle, promotion & appraisal criteria, code of conduct, anti-harassment policy, grievance process, remote work policy, training & development.

**Finance** (12 documents): travel per diem, expense reimbursement, relocation reimbursement, corporate card policy, petty cash, vendor payment terms, invoice timeline, budget approval workflow, asset/equipment purchase approval, procurement approval by role, tax documentation, year-end closing.

**Legal** (12 documents): contract approval workflow, data breach reporting, data privacy policy, data retention, employee data handling, IP policy, anti-bribery policy, conflict of interest, regulatory compliance overview, terms of service summary, third-party audits, vendor compliance.

**Support** (12 documents), by product line:
- **Numi 2.0** (smart toilet) — overview, Alexa/voice control troubleshooting
- **Leap collection** (entry-level toilets) — collection comparison, bidet wand troubleshooting
- **Anthem** (shower system) — EvoCycle setup, cycle-mode water savings
- **Verdera Voice** (smart mirror) — setup, lighting issues
- **Konnect app** — device pairing issues
- Cross-product: returns & exchanges, smart product warranty policy, installation support contact