import os
import sqlite3
import warnings
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================
# INITIALIZATION
# ============================================================
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

if not os.environ.get("GROQ_API_KEY"):
    print("\n" + "!" * 60)
    print("  ERROR: GROQ_API_KEY is missing.")
    print("  Get a free key at: https://console.groq.com")
    print("!" * 60 + "\n")
    exit(1)

ai_brain = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
DB_FILE  = "medical.db"

current_patient: Dict[str, Any] = {"name": None}

# ============================================================
# DATABASE HELPER
# ============================================================
def query_db(sql: str, params: tuple = (), write: bool = False) -> List[Any]:
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            if write:
                conn.commit()
                return []
            return cur.fetchall()
    except sqlite3.Error as e:
        print(f"  [DB Error] {e}")
        return []

# ============================================================
# SYMPTOM NORMALIZATION
# ============================================================
def normalize_symptoms(text: str) -> str:
    text = text.lower().strip()

    # Hindi → English (from DB)
    rows = query_db("SELECT hindi_word, english_word FROM hindi_map")
    for hindi, english in rows:
        text = text.replace(hindi.lower(), english.lower())

    # Common aliases
    aliases = {
        "head hurts":          "headache",
        "head pain":           "headache",
        "feel like vomiting":  "nausea",
        "high temperature":    "fever",
        "temperature":         "fever",
        "dry cough":           "cough",
        "wet cough":           "cough",
        "shortness of breath": "breathing difficulty",
        "cant breathe":        "breathing difficulty",
        "heart pain":          "chest pain",
        "blocked nose":        "runny nose",
        "stuffy nose":         "runny nose",
        "feel tired":          "fatigue",
        "feel dizzy":          "dizziness",
        "stomach ache":        "stomach pain",
        "tummy pain":          "stomach pain",
        "loose motion":        "diarrhea",
        "joint ache":          "joint pain",
        "itchy skin":          "itching",
        "blurry vision":       "blurred vision",
        "cant sleep":          "insomnia",
        "yellow eyes":         "jaundice",
        "yellow skin":         "jaundice",
        "piliya":              "jaundice",
    }
    for phrase, keyword in aliases.items():
        text = text.replace(phrase, keyword)

    return text

# ============================================================
# DIAGNOSIS
# ============================================================
def diagnose_symptoms(user_symptoms: str) -> Dict | None:
    normalized   = normalize_symptoms(user_symptoms)
    input_words  = [w for w in normalized.replace(",", " ").split() if len(w) > 2]

    if not input_words:
        return None

    variations = query_db(
        "SELECT disease, category, severity, symptoms_reported, all_symptoms, "
        "medicine, consultation, specialist FROM disease_variations"
    )
    master = query_db(
        "SELECT disease, category, severity, symptoms, symptoms, "
        "medicine, consultation, specialist FROM diseases"
    )

    best, top = None, 0
    for dis, cat, sev, s1, s2, med, consult, spec in (variations + master):
        combined = f"{s1 or ''} {s2 or ''}".lower()
        score    = sum(w in combined for w in input_words)
        if score > top:
            top  = score
            best = dict(disease=dis, category=cat, severity=sev,
                        medicine=med, consultation=consult, specialist=spec, score=score)

    return best if top > 0 else None

# ============================================================
# PATIENT HISTORY
# ============================================================
def save_history(name: str, symptoms: str, result: Dict) -> None:
    query_db(
        "INSERT INTO patient_history (username, symptoms, disease, medicine, severity, specialist) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (name, symptoms, result["disease"], result["medicine"],
         result["severity"], result["specialist"]),
        write=True
    )

def show_history(name: str) -> str:
    rows = query_db(
        "SELECT symptoms, disease, medicine, severity, specialist "
        "FROM patient_history WHERE username=? ORDER BY id DESC",
        (name,)
    )
    if not rows:
        return f"No records found for '{name}'."

    lines = [f"\n  Medical history for {name} ({len(rows)} record(s)):\n"]
    sep   = "  " + "-" * 50
    for i, (sym, dis, med, sev, spec) in enumerate(rows, 1):
        lines += [
            sep,
            f"  Record #{i}",
            f"    Disease   : {dis}  [{sev}]",
            f"    Symptoms  : {sym}",
            f"    Medicines : {med}",
            f"    Specialist: {spec}",
        ]
    lines.append(sep)
    return "\n".join(lines)

# ============================================================
# AI RESPONSE BUILDER
# ============================================================
SYSTEM_PROMPT = """You are MedAssist AI, a professional and empathetic medical assistant.

Rules:
- Respond ONLY in clear, simple English regardless of what language the user writes in.
- Use the structured diagnosis result below to explain the condition to the patient.
- Cover: suspected condition, severity, recommended medicines, specialist to see, and key advice.
- If severity is CRITICAL or HIGH, urge the patient to seek emergency care immediately.
- Keep the response concise — under 150 words.
- End every response with exactly this line:
  ⚠️  Disclaimer: This is not a formal medical diagnosis. Please consult a licensed doctor.

Diagnosis data:
{diagnosis_block}
"""

def get_ai_response(user_input: str, diagnosis: Dict) -> str:
    block = (
        f"Disease   : {diagnosis['disease']}\n"
        f"Category  : {diagnosis['category']}\n"
        f"Severity  : {diagnosis['severity']}\n"
        f"Medicines : {diagnosis['medicine']}\n"
        f"Specialist: {diagnosis['specialist']}\n"
        f"Advice    : {diagnosis['consultation']}"
    )
    prompt = SYSTEM_PROMPT.format(diagnosis_block=block)
    resp   = ai_brain.invoke([("system", prompt), ("human", user_input)])
    return resp.content

# ============================================================
# DISPLAY HELPERS
# ============================================================
SEVERITY_ICONS = {"MILD": "🟢", "MODERATE": "🟡", "CRITICAL": "🔴"}

def print_header():
    print("\n" + "=" * 58)
    print("        MedAssist AI  —  Medical Assistant")
    print("=" * 58)
    print("  Commands:")
    print("    my name is <name>   Set your patient profile")
    print("    show my history     View past diagnosis records")
    print("    clear               Clear screen")
    print("    quit                Exit")
    print("=" * 58 + "\n")

def print_diagnosis_card(result: Dict):
    sev_icon = SEVERITY_ICONS.get(result["severity"].upper(), "⚪")
    meds     = ", ".join(m.strip() for m in result["medicine"].split(",") if m.strip())
    sep      = "  " + "─" * 50

    print(f"\n{sep}")
    print(f"  Diagnosis Result")
    print(sep)
    print(f"  Condition  : {result['disease']}")
    print(f"  Category   : {result['category'].replace('_', ' ').title()}")
    print(f"  Severity   : {sev_icon}  {result['severity']}")
    print(f"  Specialist : {result['specialist']}")
    print(f"  Medicines  : {meds}")
    print(f"  Symptoms matched: {result['score']}")
    print(sep)

# ============================================================
# MAIN LOOP
# ============================================================
if __name__ == "__main__":
    print_header()

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            low = user_input.lower()

            # ── Exit ──────────────────────────────────────
            if low in ("quit", "exit", "bye"):
                print("\n  MedAssist AI: Take care! Goodbye.\n")
                break

            # ── Clear screen ──────────────────────────────
            if low == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                print_header()
                continue

            # ── Set patient name ──────────────────────────
            if low.startswith("my name is"):
                name = user_input[10:].strip().title()
                if name:
                    current_patient["name"] = name
                    print(f"\n  MedAssist AI: Hello, {name}! Your profile is now active.\n")
                else:
                    print("\n  MedAssist AI: Please provide a name after 'my name is'.\n")
                continue

            # ── Show history ──────────────────────────────
            if "show my history" in low or "medical records" in low:
                if not current_patient["name"]:
                    print("\n  MedAssist AI: Please set your name first — e.g. 'my name is Alex'.\n")
                else:
                    print(show_history(current_patient["name"]))
                continue

            # ── Diagnose ──────────────────────────────────
            print("\n  MedAssist AI: Analysing symptoms...\n")
            result = diagnose_symptoms(user_input)

            if not result:
                print(
                    "  MedAssist AI: No matching condition found in our database.\n"
                    "  Try describing more symptoms, or visit a General Physician.\n"
                )
                continue

            # Save history if patient is identified
            if current_patient["name"]:
                save_history(current_patient["name"], user_input, result)

            # Print structured card
            print_diagnosis_card(result)

            # Get & print AI explanation
            print("  AI Advice:\n")
            ai_reply = get_ai_response(user_input, result)
            for line in ai_reply.splitlines():
                print(f"  {line}")
            print()

        except KeyboardInterrupt:
            print("\n\n  MedAssist AI: Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n  [Error] {e}\n")