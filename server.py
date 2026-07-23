#!/usr/bin/env python3
"""
Resonance-engine MVP (from Chat-Bot.pdf spec).

Architecture per the document:
  - CODE controls: state machine, confidence updates, claim routing,
    option->hypothesis mapping, branch selection.
  - LLM does only: evidence extraction, reaction classification,
    hypothesis generation, constrained language rendering.

Zero dependencies (stdlib only). Needs GEMINI_API_KEY in env.
Run:  GEMINI_API_KEY=... python3 server.py   then open http://localhost:8420
"""
import json
import os
import re
import time
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

API_KEY = os.environ.get("GEMINI_API_KEY", "")
ANALYZE_MODEL = os.environ.get("ANALYZE_MODEL", "gemini-3-flash-preview")
RENDER_MODEL = os.environ.get("RENDER_MODEL", "gemini-3.1-pro-preview")
GENERIC_MODEL = os.environ.get("GENERIC_MODEL", "gemini-2.5-flash")
PORT = int(os.environ.get("PORT", "8420"))
BASE = Path(__file__).parent

# ---------------------------------------------------------------- LLM client

def gemini(model, prompt, json_mode=True, temperature=0.7, retries=2):
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(),
        headers={"x-goog-api-key": API_KEY, "Content-Type": "application/json"},
    )
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                data = json.loads(r.read())
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if "text" in p)
            if text.strip():
                return text.strip()
            last = RuntimeError("empty model response")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise last


def parse_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None
    return None


# ---------------------------------------------------------------- Prompts

ANALYZE_PROMPT = """You are the ANALYSIS module of a resonance engine. You never chat with the user. You output strict JSON only.

CURRENT ENGINE STATE
- Dialogue state: {state}
- Episode model: {episode}
- Hypotheses (keep existing ids stable; refine fields as evidence grows; you may add new ones starting at {next_hid}; set status "rejected" only when the user clearly contradicts one):
{hypotheses}
- Evidence ledger: {evidence}
- Last bot message: {last_bot}
- Options shown with it: {options}

RECENT CONVERSATION
{history}

NEW USER MESSAGE:
\"\"\"{message}\"\"\"

OUTPUT exactly this JSON shape:
{{
  "reaction": "strong_confirmation|partial_confirmation|correction|rejection|resistance|deflection|uncertainty|elaboration|action_request|new_topic|first_message",
  "reaction_note": "what specifically was confirmed, corrected or rejected (empty if none)",
  "option_match": null,
  "new_evidence": [
    {{"type": "direct_statement|reported_behaviour|linguistic_signal|repeated_theme|contradiction",
      "content": "short concrete fact, close to the user's words",
      "supports": ["H1"], "contradicts": []}}
  ],
  "episode": {{"domain": "", "surface_question": "", "trigger_type": "", "temporal_stage": "", "duration": "", "ambiguity": "", "stakes": "", "desired_outcome": "", "people": ""}},
  "hypotheses": [
    {{"id": "H1", "latent_question": "", "appraisal": "", "emotion": [""], "needs": [""], "threats": [""],
      "tension": "", "strategy": "", "reward": "", "cost": "", "threatened_identity": "",
      "fork_label": "4-7 word distinguishing concern", "status": "open"}}
  ]
}}

RULES
1. reaction describes how the user responded to the LAST BOT MESSAGE (not to life). If there was no prior bot message, use "first_message".
2. option_match: if options were shown and the user's free-text reply picks one, put its 0-based index; else null.
3. Always carry forward every still-plausible existing hypothesis (same id). Add new ones only if a genuinely different causal reading appears. Total 3-4 open hypotheses max.
4. Hypotheses must COMPETE: they explain the same evidence with different threats/needs. On a NEW episode (fewer than 2 existing open hypotheses on this topic) ALWAYS return 3-4 competing hypotheses. Canonical example - "When will I get married?" should spawn: falling behind (timing/belonging threat), fear of choosing wrongly (judgment threat), losing freedom (autonomy threat), not being chosen (worth threat). Later turns: carry forward + refine, still 3-4 while anything is undecided.
5. Stay ONE abstraction level above the evidence. Inconsistent replies -> uncertainty, never childhood wounds. No origin claims. No invented biography. No certainty about another person's mind.
6. fork_label is what the user would tap to say "it's this one": a concrete life-situation phrase they instantly recognize, e.g. "seeing friends move ahead", "family keeps asking about it", "afraid of choosing the wrong person", "losing my freedom". NEVER meta options about the bot or conversation itself ("testing what you can do", "just curious", "something else") unless the user explicitly says so.
6b. EVERY hypothesis field must be specific, never placeholder:
   - tension: a named pair like "hope<->self-protection", "autonomy<->attachment", "certainty<->possibility", "security<->freedom", "duty<->desire", "authenticity<->belonging", "control<->surrender". Never "moderate"/"high"/"N/A".
   - strategy: an observable behaviour (comparison, reassurance seeking, withdrawal, overanalysis, people pleasing, delaying commitment, checking).
   - reward / cost: concrete short-term payoff and delayed price of that strategy.
   - threatened_identity: a self-belief in the user's voice, e.g. "someone whose life is on track", "someone who is chosen", "someone who makes good decisions".
7. DRILL DOWN after a selection: once the user picks a fork, the next fork_labels must separate sub-readings of the SELECTED hypothesis, not repeat the old split. Example: after "waiting longer than expected" is picked, the next fork separates its sources: "my own timeline", "family pressure", "watching others move ahead".
8. episode: fill only fields supported by evidence; keep prior values unless contradicted; empty string if unknown.
9. new_evidence: only facts from THIS message. If the message is pure agreement/selection, return []. supports/contradicts reference hypothesis ids. Never emit evidence of type user_selected_option - the engine logs selections itself.
"""

RENDER_PROMPT = """You are the LANGUAGE RENDERER of a resonance engine. You phrase an approved plan. You must NOT invent psychology, facts, events, biography, advice domains, or interpretations beyond APPROVED CONTENT below.

DIALOGUE ACT: {act}
CERTAINTY REGISTER: {band}
- low: hedge openly ("one possibility is..."), offering alternatives is required
- medium: provisional ("it sounds as though...", "maybe")
- high: calm pattern language ("the pattern seems to be...", "from what you've said") - still owned as an interpretation, not fact
ALLOWED CLAIM TYPES: {claims}
TONE: {tone} (gentle = soft edges; direct = plain and unvarnished; blunt = say the hard thing kindly but squarely. Never cruel, never clinical jargon.)
LANGUAGE: reply in the SAME language and register as the user's latest message - if they write Hinglish, reply in natural Hinglish (including all option chips); if Hindi, reply in Hindi; if English, simple English. Never switch to stiff formal English when the user is writing casually.

APPROVED CONTENT (the ONLY material you may use):
{plan}

RECENT CONVERSATION (for continuity of phrasing only):
{history}

OUTPUT JSON:
{{"message": "...", "options": [{{"label": "...", "kind": "hypothesis_choice"}}, ...]}}

ACT RULES
- differentiate: 1-2 sentences mirroring what the user ACTUALLY said (show them they were heard, use their specifics). Then ONE forced-choice question that separates the competing readings. options: 2-4 short phrases, one per fork label in the plan, in the SAME ORDER, each kind "hypothesis_choice". Example shape: "Is the heavier part X, Y, or Z?" NEVER offer an option whose meaning repeats anything in forbidden_repeat_labels - those were already picked or declined.
- mirror: reflect the user's own specifics back (paraphrase 1-2 concrete things they actually said), then name the tentative need/threat from the top hypothesis as a PROVISIONAL interpretation. Then ONE either/or calibration question contrasting the two calibration_alternatives in the plan. The two sides must be genuinely different concerns, never two phrasings of the same thing. options: those two alternatives as 2 chips, kind "hypothesis_choice".
- synthesis: compile 2-4 of the approved components in this order of precedence: mirror of their specifics, the tension (both sides legitimate), the protective purpose of their strategy (dignify it), its cost, the threatened identity. Then ONE falsifiable question like "Which part of this is off?". options: [{{"label":"That's exactly it","kind":"reaction"}},{{"label":"Partly","kind":"reaction"}},{{"label":"Not really","kind":"reaction"}}]
- agency: briefly separate what is known / feared / unknown, then offer ONE observable, reversible step the user can take this week that serves their desired_outcome (a conversation to have, a boundary to test, a decision to schedule, or one concrete thing to watch for). Make it specific to their situation, not a generic journaling prompt. Only if nothing actionable fits, offer one concrete question to sit with. Do not deepen. options: []
- agency with is_recap=true: do NOT propose a new step. Briefly recap what has been confirmed (2-3 sentences max), restate the one step already given, and close warmly. This is the resolution, not another analysis round.

HARD RULES
- Max {max_sentences} sentences before any question. No bullet lists. No headers. No emoji. No "As an AI". No diagnosis. No predictions of outcomes/dates.
- Every factual clause must come from APPROVED CONTENT or the user's own words in the conversation.
- Do not introduce a new person, event, cause, or childhood origin.
- Never ask a circular question where both sides mean the same thing (e.g. "when" vs "why" of the same fact).
- Do not start consecutive replies with the same opening phrase (e.g. "From what you've said") - vary naturally.
- Return ONLY the JSON object.
"""

GENERIC_PROMPT = """You are a warm, supportive, generic AI companion. Respond to the user's latest message with empathy, validation and gentle encouragement. 2-4 sentences. No structure, no analysis shown.

Conversation so far:
{history}

User's latest message: {message}
Your reply:"""


# ---------------------------------------------------------------- Session state

class Session:
    def __init__(self):
        self.state = "ORIENT"
        self.turn = 0
        self.evidence = []          # [{id,type,content,confidence,supports,contradicts}]
        self.episode = {}
        self.hypotheses = []        # [{id,...,confidence,status}]
        self.history = []           # [(role, text)]
        self.generic_history = []   # [(role, text)]
        self.last_bot = ""
        self.last_options = []      # [{label, kind, hyp_id}]
        self.forks_asked = 0
        self.mirrored = False
        self.synthesized = False
        self.resistance_seen = False
        self.last_reaction = "first_message"
        self.reaction_note = ""
        self.e_counter = 0
        self.h_counter = 0
        self.validation_note = ""
        self.synthesis_count = 0
        self.synth_evidence_count = 0
        self.last_fork_hyps = []
        self.taken_fork_labels = []   # labels already shown, never re-offer
        self.agency_count = 0

    def open_hyps(self):
        return sorted([h for h in self.hypotheses if h.get("status") == "open"],
                      key=lambda h: -h.get("confidence", 0))

    def next_hid(self):
        return f"H{self.h_counter + 1}"

    def add_evidence(self, type_, content, supports=None, contradicts=None, confidence=1.0):
        self.e_counter += 1
        e = {"id": f"E{self.e_counter}", "type": type_, "content": content,
             "confidence": confidence, "supports": supports or [], "contradicts": contradicts or []}
        self.evidence.append(e)
        return e

    def bump(self, hid, delta):
        for h in self.hypotheses:
            if h["id"] == hid and h.get("status") == "open":
                h["confidence"] = max(0.05, min(0.97, round(h.get("confidence", 0.35) + delta, 3)))


SESSIONS = {}


def get_session(sid):
    if sid not in SESSIONS:
        SESSIONS[sid] = Session()
    return SESSIONS[sid]


# ---------------------------------------------------------------- Pipeline step 1: ANALYZE (LLM)

def analyze(s, message):
    hist = "\n".join(f"{r}: {t}" for r, t in s.history[-8:])
    prompt = ANALYZE_PROMPT.format(
        state=s.state,
        episode=json.dumps(s.episode, ensure_ascii=False),
        hypotheses=json.dumps([{k: v for k, v in h.items() if k != "confidence"} for h in s.hypotheses], ensure_ascii=False),
        evidence=json.dumps(s.evidence[-12:], ensure_ascii=False),
        last_bot=s.last_bot or "(none)",
        options=json.dumps([o["label"] for o in s.last_options], ensure_ascii=False) if s.last_options else "(none)",
        history=hist or "(none)",
        next_hid=s.next_hid(),
        message=message,
    )
    if s.taken_fork_labels:
        prompt += ("\nALREADY-OFFERED fork labels (the user has seen these; never reuse or reword them - "
                   "new fork_labels must be genuinely deeper or different): "
                   + json.dumps(s.taken_fork_labels[-12:], ensure_ascii=False) + "\n")
    raw = gemini(ANALYZE_MODEL, prompt, json_mode=True, temperature=0.4)
    data = parse_json(raw) or {}
    hyps = [h for h in (data.get("hypotheses") or []) if h.get("status") != "rejected"]
    if len(hyps) < 2 and len(s.open_hyps()) < 2:
        # hypothesis generation collapsed; demand the competition the doc requires
        raw = gemini(ANALYZE_MODEL, prompt + "\nIMPORTANT: you returned fewer than 2 hypotheses. Return 3-4 COMPETING hypotheses with different threats/needs, all fields specific.\n",
                     json_mode=True, temperature=0.5)
        data = parse_json(raw) or data
    return data


# ---------------------------------------------------------------- Pipeline step 2: UPDATE (code, deterministic)

def apply_selection(s, hyp_id_selected, shown_hyp_ids, weight=0.15):
    if hyp_id_selected:
        s.add_evidence("user_selected_option",
                       f"User selected: {hyp_id_selected}",
                       supports=[hyp_id_selected], confidence=0.9)
        s.bump(hyp_id_selected, weight)
    for hid in shown_hyp_ids:
        if hid != hyp_id_selected:
            s.bump(hid, -0.06)


PLACEHOLDER_TENSIONS = {"", "high", "moderate", "low", "high tension", "moderate tension",
                        "low tension", "n/a", "none", "unknown", "tension", "yes"}


def apply_analysis(s, data):
    reaction = data.get("reaction") or "elaboration"
    s.last_reaction = reaction
    s.reaction_note = data.get("reaction_note", "") or ""

    # episode merge
    ep = data.get("episode") or {}
    for k, v in ep.items():
        if v:
            s.episode[k] = v

    # hypotheses: merge by id, add new
    incoming = data.get("hypotheses") or []
    by_id = {h["id"]: h for h in s.hypotheses}
    for h in incoming:
        hid = h.get("id")
        if not hid:
            continue
        if str(h.get("tension", "")).strip().lower() in PLACEHOLDER_TENSIONS:
            h.pop("tension", None)  # don't let placeholder text overwrite a real tension pair
        if hid in by_id:
            old = by_id[hid]
            conf = old.get("confidence", 0.35)
            old.update({k: v for k, v in h.items() if v not in (None, "", [])})
            old["confidence"] = conf
            # honor LLM rejections only on explicit user rejection/correction; a topic
            # switch parks old hypotheses as dormant instead of declaring them false
            if h.get("status") == "rejected":
                if reaction == "new_topic":
                    old["status"] = "dormant"
                elif reaction in ("rejection", "correction"):
                    old["status"] = "rejected"
        else:
            h["confidence"] = 0.35
            h["status"] = h.get("status") or "open"
            s.hypotheses.append(h)
            m = re.match(r"H(\d+)$", hid)
            if m:
                s.h_counter = max(s.h_counter, int(m.group(1)))

    # new evidence
    for e in (data.get("new_evidence") or [])[:4]:
        if not e.get("content") or e.get("type") == "user_selected_option":
            continue
        s.add_evidence(e.get("type", "direct_statement"), e["content"],
                       supports=e.get("supports") or [], contradicts=e.get("contradicts") or [],
                       confidence=1.0 if e.get("type") == "direct_statement" else 0.7)
        for hid in (e.get("supports") or [])[:2]:
            s.bump(hid, 0.08 if e.get("type") == "direct_statement" else 0.06)
        for hid in (e.get("contradicts") or [])[:2]:
            s.bump(hid, -0.25)

    # reaction effects on top hypothesis
    top = s.open_hyps()[0] if s.open_hyps() else None
    # keep the lattice small: at most 5 open hypotheses
    for extra in s.open_hyps()[5:]:
        extra["status"] = "dormant"
    if top:
        if reaction == "strong_confirmation":
            s.bump(top["id"], 0.15)
        elif reaction == "partial_confirmation":
            pass  # partial = split the hypothesis, not endorsement; no confidence change
        elif reaction == "correction":
            s.bump(top["id"], -0.10)
        elif reaction == "rejection":
            top["status"] = "rejected"
        elif reaction == "uncertainty":
            s.bump(top["id"], -0.03)
        elif reaction == "resistance":
            s.resistance_seen = True

    # option_match (free-text answer to a fork)
    om = data.get("option_match")
    if isinstance(om, int) and 0 <= om < len(s.last_options):
        opt = s.last_options[om]
        if opt.get("kind") == "hypothesis_choice":
            apply_selection(s, opt.get("hyp_id"), [o.get("hyp_id") for o in s.last_options], weight=0.12)
        elif opt.get("kind") == "reaction":
            lab = opt["label"].lower()
            if "exactly" in lab or "fits" in lab:
                if top:
                    s.bump(top["id"], 0.15)
                s.last_reaction = "strong_confirmation"
            elif "partly" in lab:
                s.last_reaction = "partial_confirmation"  # no bump - split, don't endorse
            elif "not really" in lab or "not" in lab:
                if top:
                    top["status"] = "rejected"
                s.last_reaction = "rejection"


# ---------------------------------------------------------------- Pipeline step 3: DECIDE (code state machine)

def decide_act(s, forced_action):
    top = s.open_hyps()[0] if s.open_hyps() else None
    if forced_action or s.last_reaction == "action_request":
        return "agency", top
    if s.resistance_seen:
        # reduce depth once, then resume normal flow
        s.resistance_seen = False
        return ("mirror" if not s.mirrored else "agency"), top
    if not top:
        return "differentiate", None
    conf = top.get("confidence", 0.35)
    # partial confirmation: split the hypothesis - find out which part was off
    if s.last_reaction == "partial_confirmation" and s.forks_asked < 4:
        return "differentiate", top
    if s.state == "ORIENT":
        return "differentiate", top
    # not enough separation yet -> keep forking (drill-down); doc cadence: ~2 forks before mirror
    if conf < 0.60 and s.forks_asked < 3:
        return "differentiate", top
    if not s.mirrored:
        return "mirror", top
    if conf >= 0.68 and not s.synthesized:
        return "synthesis", top
    if s.synthesized:
        # user confirmed the synthesis -> resolve; new material -> revise once more
        if s.last_reaction == "strong_confirmation":
            return "agency", top
        if (s.last_reaction in ("elaboration", "partial_confirmation")
                and len(s.evidence) > s.synth_evidence_count and s.synthesis_count < 2):
            s.synthesized = False
            return "synthesis", top
        return "agency", top
    # mirrored but confirmation still weak -> one more fork if allowed, else provisional synthesis
    if s.forks_asked < 3:
        return "differentiate", top
    return "synthesis", top


CLAIM_BANDS = {
    "low": ["observation", "alternative_hypothesis", "calibration_question"],
    "medium": ["observation", "alternative_hypothesis", "calibration_question",
               "need_interpretation", "provisional_tension", "protective_function"],
    "high": ["observation", "calibration_question", "need_interpretation", "tension_claim",
             "protective_function", "cost_interpretation", "causal_loop",
             "threatened_identity", "peak_synthesis", "action_bridge"],
}


def band_for(conf):
    if conf < 0.50:
        return "low"
    if conf < 0.68:
        return "medium"
    return "high"


STATE_OF_ACT = {"differentiate": "DIFFERENTIATE", "mirror": "MIRROR",
                "synthesis": "SYNTHESIS", "agency": "AGENCY"}


# ---------------------------------------------------------------- Pipeline step 4+5: ROUTE + RENDER (LLM, constrained)

def render(s, act, top, tone, retry_note=""):
    conf = top.get("confidence", 0.35) if top else 0.2
    band = band_for(conf)
    claims = CLAIM_BANDS[band]
    hyps = s.open_hyps()

    plan = {"user_said_recently": [t for r, t in s.history[-4:] if r == "user"]}
    if act == "differentiate":
        plan["fork_labels"] = [{"hyp_id": h["id"], "label": h.get("fork_label") or h.get("latent_question", "")}
                               for h in hyps[:3]]
        plan["surface_question"] = s.episode.get("surface_question", "")
        plan["forbidden_repeat_labels"] = s.taken_fork_labels[-12:]
        plan["split_context"] = ("The user said the last interpretation was only PARTLY right - "
                                 "fork to find which part was off.") if s.last_reaction == "partial_confirmation" else ""
        s.last_fork_hyps = [h["id"] for h in hyps[:3]]
    elif act == "mirror":
        plan["top_hypothesis"] = {k: top.get(k) for k in
                                  ("latent_question", "appraisal", "emotion", "needs", "threats", "strategy")} if top else {}
        plan["confirmed_facts"] = [e["content"] for e in s.evidence[-5:]]
        top_threat = (top.get("threats") or [top.get("fork_label", "")])[0] if top else ""
        runner = hyps[1].get("fork_label") if len(hyps) > 1 else ""
        plan["calibration_alternatives"] = [x for x in (top_threat, runner) if x]
        s.last_fork_hyps = [h["id"] for h in hyps[:2]]
    elif act == "synthesis":
        comps = {"mirror_facts": [e["content"] for e in s.evidence[-5:]]}
        if top:
            comps.update({k: top.get(k) for k in
                          ("latent_question", "appraisal", "tension", "strategy", "reward", "cost", "threatened_identity")})
        plan["approved_components"] = comps
    elif act == "agency":
        plan["known"] = [e["content"] for e in s.evidence if e["type"] in ("direct_statement", "user_selected_option")][-5:]
        plan["feared"] = (top or {}).get("threats", [])
        plan["tension"] = (top or {}).get("tension", "")
        plan["strategy_and_cost"] = {"strategy": (top or {}).get("strategy", ""), "cost": (top or {}).get("cost", "")}
        plan["desired_outcome"] = s.episode.get("desired_outcome", "")
        plan["top_latent_question"] = (top or {}).get("latent_question", "")
        plan["is_recap"] = s.agency_count >= 1  # advice already given once: recap, don't re-prescribe

    hist = "\n".join(f"{r}: {t}" for r, t in s.history[-6:])
    prompt = RENDER_PROMPT.format(
        act=act, band=band, claims=", ".join(claims), tone=tone,
        plan=json.dumps(plan, ensure_ascii=False, indent=1),
        history=hist or "(none)",
        max_sentences=5 if act != "synthesis" else 7,
    )
    if retry_note:
        prompt += f"\nPREVIOUS ATTEMPT FAILED VALIDATION: {retry_note}. Fix it.\n"
    raw = gemini(RENDER_MODEL, prompt, json_mode=True, temperature=0.75)
    return parse_json(raw) or {}


def validate(s, act, out):
    msg = (out or {}).get("message", "")
    if not msg or len(msg) < 10:
        return "empty message"
    if len(msg) > 1100:
        return "message too long"
    opts = out.get("options")
    if act == "differentiate":
        if not isinstance(opts, list) or not (2 <= len(opts) <= 4):
            return "differentiate needs 2-4 options"
    sentences = len([x for x in re.split(r"[.!?]+\s", msg) if x.strip()])
    if sentences > 8:
        return "too many sentences"
    return None


def fallback_render(s, act, top):
    """Code-side last resort: never let the user face a parser-error shrug."""
    hyps = s.open_hyps()
    said = s.history[-1][1] if s.history else ""
    if act == "differentiate":
        labels = [h.get("fork_label") or h.get("latent_question", "?") for h in hyps[:3]]
        if len(labels) < 2:
            labels.append("something else")
        q = ", ".join(labels[:-1]) + f", or {labels[-1]}"
        msg = f'I hear you - "{said[:120]}". To make sure I read this right: is it mostly about {q}?'
        return {"message": msg, "options": [{"label": l, "kind": "hypothesis_choice"} for l in labels]}
    if act == "mirror" and top:
        return {"message": (f"It sounds as though underneath this is the question: {top.get('latent_question', '').lower()} "
                            f"Does that fit, or am I reading it wrong?"),
                "options": [{"label": "That fits", "kind": "reaction"},
                            {"label": "Partly", "kind": "reaction"},
                            {"label": "Not really", "kind": "reaction"}]}
    if act == "synthesis" and top:
        parts = [f"From what you've told me: {top.get('appraisal', '')}."]
        if top.get("tension"):
            parts.append(f"There seems to be a real tension here: {top['tension']}.")
        if top.get("strategy") and top.get("cost"):
            parts.append(f"Coping by {top['strategy']} makes sense, but it may be costing you: {top['cost']}.")
        parts.append("Which part of this is off?")
        return {"message": " ".join(parts),
                "options": [{"label": "That's exactly it", "kind": "reaction"},
                            {"label": "Partly", "kind": "reaction"},
                            {"label": "Not really", "kind": "reaction"}]}
    # agency
    known = [e["content"] for e in s.evidence if e["type"] in ("direct_statement", "user_selected_option")][-3:]
    msg = "What we know: " + "; ".join(known) + "." if known else "Here's where we've landed. "
    msg += (" The one thing I'd suggest this week: pick the smallest decision in this situation "
            "that you can reverse, and take it - then watch what actually happens.")
    return {"message": msg, "options": []}


# ---------------------------------------------------------------- Chat orchestration

def handle_chat(sid, message, tone="gentle", option=None):
    s = get_session(sid)
    s.turn += 1
    s.validation_note = ""
    forced_action = False

    # ---- step 1+2: analyze & update (deterministic application)
    if option is not None:
        # a UI chip was clicked: code applies it deterministically
        kind = option.get("kind")
        shown_ids = [o.get("hyp_id") for o in s.last_options]
        display = option.get("label", message)
        if kind == "hypothesis_choice":
            apply_selection(s, option.get("hyp_id"), shown_ids, weight=0.15)
            s.last_reaction = "elaboration"
            # re-analyze so the next fork drills down into the selected hypothesis
            saved_opts, s.last_options = s.last_options, []
            try:
                data = analyze(s, f'The user selected this option: "{display}"')
                data["new_evidence"] = []  # selection itself was already logged in code
                data["reaction"] = "elaboration"  # a selection is differentiation, not confirmation
                for h in (data.get("hypotheses") or []):
                    if h.get("status") == "rejected":
                        h["status"] = "open"  # picking one fork rejects nothing; code demotes
                apply_analysis(s, data)
                s.last_reaction = "elaboration"
            except Exception as e:  # noqa: BLE001
                s.validation_note = f"analyze_error: {e}"
            s.last_options = saved_opts
        else:  # reaction chip
            lab = (option.get("label") or "").lower()
            top = s.open_hyps()[0] if s.open_hyps() else None
            if "exactly" in lab or "fits" in lab:
                s.last_reaction = "strong_confirmation"
                if top:
                    s.bump(top["id"], 0.15)
            elif "partly" in lab:
                s.last_reaction = "partial_confirmation"  # no bump - code will ask which part is off
            else:
                s.last_reaction = "rejection"
                if top:
                    top["status"] = "rejected"
    else:
        display = message
        try:
            data = analyze(s, message)
            apply_analysis(s, data)
        except Exception as e:  # noqa: BLE001
            s.validation_note = f"analyze_error: {e}"
        low = message.lower()
        if any(p in low for p in ("what should i do", "just tell me", "what do i do",
                                  "give me advice", "what's the answer", "what is the answer")):
            forced_action = True

    s.history.append(("user", display))

    # ---- step 3: decide act (code)
    act, top = decide_act(s, forced_action)
    s.state = STATE_OF_ACT[act]
    if act == "differentiate":
        s.forks_asked += 1
    elif act == "mirror":
        s.mirrored = True
    elif act == "synthesis":
        s.synthesized = True
        s.synthesis_count += 1
        s.synth_evidence_count = len(s.evidence)
    elif act == "agency":
        s.agency_count += 1

    # ---- step 4+5: render (LLM, constrained) with one validation retry, then code fallback
    out = render(s, act, top, tone)
    bad = validate(s, act, out)
    if bad:
        s.validation_note = f"retry: {bad}"
        out = render(s, act, top, tone, retry_note=bad)
        bad2 = validate(s, act, out)
        if bad2:
            out = fallback_render(s, act, top)
            s.validation_note = f"fallback_used: {bad2}"

    reply = out.get("message") or "I'm having trouble phrasing that - could you say a bit more?"
    options = []
    if act in ("differentiate", "mirror"):
        fork_ids = s.last_fork_hyps
        for i, o in enumerate((out.get("options") or [])[:4]):
            hyp_id = fork_ids[i] if i < len(fork_ids) else None
            options.append({"label": o.get("label", "?"), "kind": "hypothesis_choice", "hyp_id": hyp_id})
    elif act == "synthesis":
        for o in (out.get("options") or []):
            options.append({"label": o.get("label", "?"), "kind": "reaction"})
    s.last_bot = reply
    s.last_options = options
    if act in ("differentiate", "mirror"):
        s.taken_fork_labels.extend(o["label"] for o in options)
    s.history.append(("assistant", reply))

    return {"reply": reply, "options": options, "debug": debug_view(s, act)}


def debug_view(s, act):
    return {
        "state": s.state, "act": act, "turn": s.turn,
        "reaction": s.last_reaction, "reaction_note": s.reaction_note,
        "forks_asked": s.forks_asked, "episode": s.episode,
        "hypotheses": sorted(s.hypotheses, key=lambda h: -h.get("confidence", 0)),
        "evidence": s.evidence[-15:],
        "validation": s.validation_note,
    }


def handle_generic(sid, message):
    s = get_session(sid)
    hist = "\n".join(f"{r}: {t}" for r, t in s.generic_history[-10:])
    prompt = GENERIC_PROMPT.format(history=hist or "(none)", message=message)
    try:
        reply = gemini(GENERIC_MODEL, prompt, json_mode=False, temperature=0.8)
    except Exception as e:  # noqa: BLE001
        reply = f"(generic bot error: {e})"
    s.generic_history.append(("user", message))
    s.generic_history.append(("assistant", reply))
    return {"generic": reply}


# ---------------------------------------------------------------- HTTP

# light abuse protection for the public deployment (per-IP message quota)
RATE = {}  # ip -> [window_start_epoch, count]
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "120"))  # messages per hour per IP


def rate_ok(ip):
    now = time.time()
    start, count = RATE.get(ip, (now, 0))
    if now - start > 3600:
        start, count = now, 0
    count += 1
    RATE[ip] = (start, count)
    return count <= RATE_LIMIT


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send(204, "")

    def _json_body(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:  # noqa: BLE001
            return {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (BASE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/healthz":
            self._send(200, '{"ok": true}')
        else:
            self._send(404, "{}")

    def do_POST(self):
        if not rate_ok(self.client_address[0]):
            self._send(429, json.dumps({"error": "rate limit - try again in an hour"}))
            return
        body = self._json_body()
        sid = body.get("session_id", "default")
        try:
            if self.path == "/chat":
                out = handle_chat(sid, body.get("message", ""), body.get("tone", "gentle"),
                                  option=body.get("option"))
            elif self.path == "/generic":
                out = handle_generic(sid, body.get("message", ""))
            elif self.path == "/reset":
                SESSIONS.pop(sid, None)
                out = {"ok": True}
            else:
                self._send(404, "{}")
                return
            self._send(200, json.dumps(out, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(e)}))

    def log_message(self, *a):  # quiet
        pass


def main():
    if not API_KEY:
        print("ERROR: set GEMINI_API_KEY first:\n  GEMINI_API_KEY=... python3 server.py")
        raise SystemExit(1)
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"resonance-bot MVP on {host}:{PORT}  (analyze={ANALYZE_MODEL}, render={RENDER_MODEL})")
    ThreadingHTTPServer((host, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
