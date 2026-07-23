# Resonance-engine chatbot — MVP hypothesis test

An MVP built from `Chat-Bot.pdf` to answer one question: **does this architecture produce a
conversations that feels genuinely different, or is it just a generic empathetic chatbot with
extra machinery?**

## Run

```bash
GEMINI_API_KEY=... python3 server.py
# open http://localhost:8420
```

Zero dependencies (Python stdlib only). Models default to `gemini-2.5-flash`; override with
`ANALYZE_MODEL` / `RENDER_MODEL` env vars.

## What the MVP implements (and what it deliberately skips)

Implemented, per the doc's separation of concerns:

- **Code (deterministic):** dialogue state machine (`ORIENT → DIFFERENTIATE → MIRROR →
  SYNTHESIS → AGENCY`), confidence scores, claim-band routing (low/medium/high certainty
  registers), option→hypothesis mapping, branch selection, one validation retry on renders.
- **LLM (narrow jobs only):** evidence extraction + reaction classification (one JSON call),
  competing-hypothesis generation with fork labels (same call), constrained rendering of an
  approved plan (second call), and the generic control bot.
- **The fork mechanic:** every reply offers 2–4 tappable options that separate the leading
  hypotheses; a click is logged as `user_selected_option` evidence and the next fork *drills
  down* into the selected reading.
- **Truth controls (MVP level):** renderer may only use approved plan content; certainty
  register follows confidence; corrections reject/demote hypotheses in code; rejection of the
  top hypothesis flips the ranking.
- **Debug panel:** live view of state, hypotheses + confidence, evidence ledger, episode
  model — so you can see whether the machinery is doing anything real.
- **Control group:** toggle "compare with generic bot" to see a vanilla supportive-chatbot
  reply to the *same* message under each engine reply.

Skipped on purpose (not needed to test the hypothesis): activation/conversion model, memory
across sessions, full truth-validator, policy gateway, Hinglish tuning, persistence.

## Test protocol (10 minutes)

1. **Canonical flow:** click the starter "When will I get married?" and just tap chips for
   3–4 turns. Watch the debug panel: forks should drill down (timeline → whose pressure →
   the fear under it), confidence should climb, and the 3rd/4th reply should be a *synthesis*
   that names a tension and a cost — not a rephrasing of your question.
2. **Compare:** read the ghosted "generic bot" box under each reply. Same input, vanilla
   empathy. This is the control; the experiment is whether the engine's replies feel
   categorically different or just differently formatted.
3. **Correct it:** when it mirrors, answer "no, it's not family, it's X". The top hypothesis
   should drop or flip in the debug panel — no semantic rescue, no "your resistance confirms it".
4. **Resist:** say "I don't want to get into that". Depth should reduce (mirror → agency),
   not push deeper.
5. **Ask for action:** "so what should I do?" — deepening should stop; you get one
   observable, reversible step, not more analysis.
6. **Try to break it:** single-word answers, topic jumps mid-flow, "you're just making stuff
   up". Note where the state machine does something dumb — that's the reorientation signal.

## What to judge

- Do the forks feel like the bot is *doing the interpretive work* (vs. making you explain
  yourself)?
- Does the synthesis produce even one "huh, I hadn't connected that" moment? (The doc's
  whole bet is recognition/revelation/relief.)
- Or does it collapse into Barnum statements with extra steps? If the debug panel shows
  hypotheses churning but the *text* still reads generic, the bottleneck is rendering, not
  architecture — that's a different fix than "the idea doesn't work".

## What the persona swarm already found (5 adversarial testers, scores 7/6/5/5/5)

Consensus: **categorically different machinery, not a generic bot** — every tester confirmed
the hypothesis tracking, drill-down forks and correction handling are real (correcting it
actually rebuilds the model; the generic control is content-free validation by comparison).
Also found (and since fixed): "Partly" treated as confirmation, circular re-forks, renderer
shrugging on emotional turns, resistance locking into advice mode, Hinglish answered in
stiff English, advice repeated with no recap. Watch for whether they stay fixed.

Known open weaknesses: confidence numbers are still vibes-with-decimals; the fork-question
phrasing template gets samey over a long session; ~10-30s latency per turn (pro render).

## Files

- `server.py` — engine + HTTP + Gemini client (single file)
- `index.html` — chat UI + debug panel + generic compare
- `test_flow.py` — scripted end-to-end run of the canonical flow (needs server running)
