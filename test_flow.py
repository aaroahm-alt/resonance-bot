#!/usr/bin/env python3
"""Scripted end-to-end test of the canonical flow from Chat-Bot.pdf:
surface question -> fork 1 -> drill-down fork 2 -> mirror -> synthesis -> agency.
Also exercises correction handling and the generic control bot."""
import json
import sys
import urllib.request

BASE = "http://localhost:8420"
SID = "scripted-1"


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def show(tag, d):
    dbg = d.get("debug", {})
    print(f"\n=== {tag} ===")
    print("BOT:", d.get("reply"))
    for o in d.get("options", []):
        print("   chip:", o["label"], f"({o.get('kind')}, {o.get('hyp_id')})")
    print("STATE:", dbg.get("state"), "| act:", dbg.get("act"), "| reaction:", dbg.get("reaction"))
    print("HYPS:", [(h["id"], round(h["confidence"], 2), h["status"],
                     (h.get("fork_label") or "")[:40]) for h in dbg.get("hypotheses", [])])
    if dbg.get("validation"):
        print("VALIDATION:", dbg["validation"])
    return d


def chip(d, idx):
    o = d["options"][idx]
    return {"label": o["label"], "kind": o["kind"], "hyp_id": o.get("hyp_id")}


def main():
    post("/reset", {"session_id": SID})

    d = post("/chat", {"session_id": SID, "message": "When will I get married?", "tone": "direct", "option": None})
    show("T1 surface question (expect DIFFERENTIATE, 3 competing forks)", d)
    assert d["options"], "expected fork options"

    g = post("/generic", {"session_id": SID, "message": "When will I get married?"})
    print("\nGENERIC CONTROL:", g["generic"])

    d = post("/chat", {"session_id": SID, "message": d["options"][0]["label"], "tone": "direct", "option": chip(d, 0)})
    show("T2 clicked fork 1 (expect drill-down DIFFERENTIATE or MIRROR)", d)

    if d["options"] and d["options"][0]["kind"] == "hypothesis_choice":
        d = post("/chat", {"session_id": SID, "message": d["options"][0]["label"], "tone": "direct", "option": chip(d, 0)})
        show("T3 clicked drill-down fork", d)

    # free-text elaboration with a concrete fact
    d = post("/chat", {"session_id": SID, "message": "My family asks about it every single week, and two of my friends got married this year. I keep catching myself comparing.", "tone": "direct", "option": None})
    show("T4 free-text elaboration (expect MIRROR or SYNTHESIS)", d)

    # confirm whatever was asked, or react to synthesis chips
    if d["options"] and d["options"][0]["kind"] == "reaction":
        d = post("/chat", {"session_id": SID, "message": "That's exactly it", "tone": "direct", "option": chip(d, 0)})
        show("T5 confirmed synthesis (expect AGENCY)", d)
    elif d["options"]:
        d = post("/chat", {"session_id": SID, "message": d["options"][0]["label"], "tone": "direct", "option": chip(d, 0)})
        show("T5 answered mirror fork", d)
        if d["options"] and d["options"][0]["kind"] == "reaction":
            d = post("/chat", {"session_id": SID, "message": "That's exactly it", "tone": "direct", "option": chip(d, 0)})
            show("T6 confirmed synthesis (expect AGENCY)", d)

    d = post("/chat", {"session_id": SID, "message": "So what should I actually do?", "tone": "direct", "option": None})
    show("T-END action request (expect AGENCY, no chips)", d)

    print("\n--- correction path (fresh session) ---")
    post("/reset", {"session_id": "scripted-2"})
    d = post("/chat", {"session_id": "scripted-2", "message": "Will he come back to me?", "tone": "gentle", "option": None})
    show("C1 surface question", d)
    d = post("/chat", {"session_id": "scripted-2", "message": "No, you're reading it wrong - I don't want him back, I want to know if I wasted two years.", "tone": "gentle", "option": None})
    show("C2 user correction (top hyp should drop/flip)", d)

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
