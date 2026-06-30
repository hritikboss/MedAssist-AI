import streamlit as st
import sqlite3
import bcrypt
import math
import re
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# ── MUST be the very first Streamlit call in the script ───────────────────
st.set_page_config(page_title="MedAssist AI", page_icon="🩺", layout="centered")

# ── Single shared AI client (not recreated every rerun) ────────────────────
@st.cache_resource
def get_ai_client():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)

ai_brain = get_ai_client()

# ============================================================
# DB  —  one persistent read-only connection for lookups
# ============================================================
@st.cache_resource
def get_read_conn():
    """Persistent read-only connection, reused across all reruns."""
    conn = sqlite3.connect("medical.db", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")   # allow concurrent reads + writes
    conn.execute("PRAGMA query_only=ON")
    return conn

def write_db(sql: str, params: tuple) -> None:
    """Short-lived connection only for INSERT/UPDATE — not kept open."""
    with sqlite3.connect("medical.db") as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(sql, params)
        conn.commit()

# ============================================================
# SYMPTOM DIARY  —  DB setup + helpers
# ============================================================
def init_diary_table():
    write_db("""CREATE TABLE IF NOT EXISTS symptom_diary (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        username   TEXT,
        date       TEXT,
        energy     INTEGER,
        pain       INTEGER,
        mood       INTEGER,
        symptoms   TEXT,
        notes      TEXT
    )""", ())

init_diary_table()

def save_diary_entry(username, date, energy, pain, mood, symptoms, notes):
    write_db(
        "INSERT INTO symptom_diary (username,date,energy,pain,mood,symptoms,notes) "
        "VALUES (?,?,?,?,?,?,?)",
        (username, date, energy, pain, mood, symptoms, notes)
    )

def get_diary_entries(username):
    return get_read_conn().execute(
        "SELECT date,energy,pain,mood,symptoms,notes FROM symptom_diary "
        "WHERE username=? ORDER BY date DESC", (username,)
    ).fetchall()

# ============================================================
# MEDICINE REMINDERS  —  DB setup + helpers
# ============================================================
def init_reminders_table():
    write_db("""CREATE TABLE IF NOT EXISTS reminders (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT,
        medicine  TEXT,
        dose      TEXT,
        time_1    TEXT,
        time_2    TEXT,
        time_3    TEXT,
        active    INTEGER DEFAULT 1
    )""", ())

init_reminders_table()

def save_reminder(username, medicine, dose, time_1, time_2, time_3):
    write_db(
        "INSERT INTO reminders (username,medicine,dose,time_1,time_2,time_3) VALUES (?,?,?,?,?,?)",
        (username, medicine, dose, time_1, time_2, time_3)
    )

def get_reminders(username):
    return get_read_conn().execute(
        "SELECT id,medicine,dose,time_1,time_2,time_3,active FROM reminders WHERE username=? AND active=1",
        (username,)
    ).fetchall()

def delete_reminder(reminder_id):
    write_db("UPDATE reminders SET active=0 WHERE id=?", (reminder_id,))

# ============================================================
# CACHED DATA  —  loaded once, never re-fetched per rerun
# ============================================================
@st.cache_data
def load_hindi_map() -> dict:
    try:
        rows = get_read_conn().execute(
            "SELECT hindi_word, english_word FROM hindi_map"
        ).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}

@st.cache_data
def load_disease_rows() -> list:
    """
    Load ALL variation + master rows once and cache them forever.
    Returns a list of (disease, category, icd_code, severity,
                        combined_symptoms, medicine, consultation, specialist,
                        age_group, gender)
    """
    conn = get_read_conn()

    variations = conn.execute("""
        SELECT disease, category, icd_code, severity,
               (COALESCE(symptoms_reported,'') || ' ' || COALESCE(all_symptoms,'')),
               medicine, consultation, specialist,
               COALESCE(age_group,''), COALESCE(gender,'')
        FROM disease_variations
    """).fetchall()

    master = conn.execute("""
        SELECT disease, category, icd_code, severity,
               (COALESCE(symptoms,'') || ' ' || COALESCE(symptoms,'')),
               medicine, consultation, specialist,
               '', ''
        FROM diseases
    """).fetchall()

    return [
        (dis, cat, icd, sev, syms.lower(), med, consult, spec, age_grp, gen)
        for dis, cat, icd, sev, syms, med, consult, spec, age_grp, gen
        in (variations + master)
    ]

@st.cache_data(ttl=30)   # sidebar stats refresh every 30 s, not every rerun
def load_sidebar_stats(username: str) -> tuple:
    conn = get_read_conn()
    try:
        d = conn.execute("SELECT COUNT(*) FROM diseases").fetchone()[0]
        v = conn.execute("SELECT COUNT(*) FROM disease_variations").fetchone()[0]
    except Exception:
        d, v = 0, 0
    # history count needs a write-conn because PRAGMA query_only blocks it
    try:
        with sqlite3.connect("medical.db") as wc:
            h = wc.execute(
                "SELECT COUNT(*) FROM patient_history WHERE username=?", (username,)
            ).fetchone()[0]
    except Exception:
        h = 0
    return d, v, h

@st.cache_data(ttl=60)   # analytics data — refresh every 60 s
def load_analytics_data(username: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect("medical.db") as conn:
        all_df = pd.read_sql_query("""
            SELECT ph.disease, ph.severity, ph.specialist,
                   COALESCE(d.category, 'unknown') AS category
            FROM   patient_history ph
            LEFT JOIN diseases d ON ph.disease = d.disease
        """, conn)
        user_df = pd.read_sql_query("""
            SELECT ph.disease, ph.severity, ph.specialist,
                   COALESCE(d.category, 'unknown') AS category
            FROM   patient_history ph
            LEFT JOIN diseases d ON ph.disease = d.disease
            WHERE  ph.username = ?
        """, conn, params=(username,))
    return all_df, user_df

# ============================================================
# ALIAS MAP  (module-level constant, never rebuilt)
# ============================================================
ALIASES: dict[str, str] = {
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
    "sad all the time":    "depression",
    "feeling anxious":     "anxiety",
    "piliya":              "jaundice",
    "yellow eyes":         "jaundice",
    "yellow skin":         "jaundice",
}

# ============================================================
# EMAIL ALERT  —  fires when severity is CRITICAL
# ============================================================
import smtplib, os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_critical_alert(to_email: str, username: str, disease: str,
                         medicine: str, specialist: str) -> None:
    """Send a CRITICAL diagnosis alert via Gmail SMTP (env-configured)."""
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    if not smtp_user or not smtp_pass or not to_email:
        return   # silently skip if not configured

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 MedAssist CRITICAL Alert — {disease}"
    msg["From"]    = smtp_user
    msg["To"]      = to_email

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:540px;margin:auto;
                border:2px solid #FF4B4B;border-radius:8px;padding:24px;">
      <h2 style="color:#FF4B4B;">🚨 CRITICAL Health Alert</h2>
      <p>Hi <b>{username}</b>, MedAssist AI has flagged a <b>CRITICAL</b> condition.</p>
      <table style="width:100%;border-collapse:collapse;margin-top:12px;">
        <tr><td style="padding:6px;font-weight:bold;">Condition</td>
            <td style="padding:6px;">{disease}</td></tr>
        <tr style="background:#FFF5F5;"><td style="padding:6px;font-weight:bold;">Medicines</td>
            <td style="padding:6px;">{medicine}</td></tr>
        <tr><td style="padding:6px;font-weight:bold;">Specialist</td>
            <td style="padding:6px;">{specialist}</td></tr>
      </table>
      <p style="margin-top:16px;color:#FF4B4B;font-weight:bold;">
        ⚠️ Please seek emergency medical care immediately.</p>
      <p style="font-size:11px;color:#888;margin-top:24px;">
        This is an AI-generated alert and not a formal medical diagnosis.</p>
    </div>"""

    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
    except Exception as e:
        st.error(f"⚠️ Email alert failed to send: {e}")

# ============================================================
# NORMALIZE SYMPTOMS  (pure function — no DB call)
# ============================================================
def normalize_symptoms(text: str, hindi_map: dict) -> str:
    text = text.lower().strip()
    for hindi, english in hindi_map.items():
        text = text.replace(hindi, english)
    for phrase, keyword in ALIASES.items():
        text = text.replace(phrase, keyword)
    return text

# ============================================================
# NEGATION DETECTION  —  strip negated symptoms before matching
# ============================================================
_NEGATION_PREFIXES = re.compile(
    r"\b(no|not|without|never|don't have|dont have|do not have|absence of)\s+(\w+)", re.I
)

def remove_negated_symptoms(text: str) -> str:
    """Remove words that are negated, e.g. 'no fever' → fever removed."""
    negated = set()
    for m in _NEGATION_PREFIXES.finditer(text):
        negated.add(m.group(2).lower())
    if not negated:
        return text
    words = text.split()
    return " ".join(w for w in words if w.lower() not in negated)

# ============================================================
# FUZZY MATCHING  —  handles typos like "hedache" → "headache"
# ============================================================
def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j] + (ca != cb), curr[j] + 1, prev[j + 1] + 1))
        prev = curr
    return prev[-1]

def fuzzy_match(word: str, symptom_text: str, threshold: int = 2) -> bool:
    """Return True if word approximately matches any token in symptom_text."""
    if word in symptom_text:          # exact match first (fast path)
        return True
    if len(word) <= 3:                # too short for fuzzy — avoid false positives
        return False
    # Scale the allowed edit distance with word length so short/medium words
    # (e.g. "viral" vs "oral") aren't treated as typos of each other.
    # 4-5 letter words: max 1 edit. 6+ letter words: up to `threshold` edits.
    max_allowed = 1 if len(word) <= 5 else threshold
    for token in symptom_text.split():
        if abs(len(token) - len(word)) > max_allowed:
            continue
        if _edit_distance(word, token) <= max_allowed:
            return True
    return False

# ============================================================
# TF-IDF SCORING  —  rare symptom words score higher
# ============================================================
@st.cache_data
def build_idf_map() -> dict[str, float]:
    """
    Compute IDF weight for every symptom token across all disease rows.
    Cached once — rebuilds only on app restart.
    """
    rows = load_disease_rows()
    N    = len(rows)
    df_count: dict[str, int] = {}
    for row in rows:
        syms = row[4]   # index 4 = combined_symptoms
        tokens = set(syms.split())
        for t in tokens:
            df_count[t] = df_count.get(t, 0) + 1
    return {t: math.log((N + 1) / (cnt + 1)) + 1 for t, cnt in df_count.items()}

@st.cache_data
def build_symptom_list() -> list[str]:
    """Extract every unique symptom token from the DB for autocomplete."""
    rows = load_disease_rows()
    tokens = set()
    for row in rows:
        syms = row[4]   # index 4 = combined_symptoms
        for token in syms.split():
            if len(token) > 3:
                tokens.add(token.strip(",."))
    return sorted(tokens)

# ============================================================
# DURATION-BASED SEVERITY UPGRADE
# ============================================================
_DURATION_WEIGHTS = {
    "just started": 0,
    "1-2 days":     0,
    "3-5 days":     1,
    "1 week":       1,
    "2 weeks":      2,
    "1 month":      2,
    "3+ months":    3,
}

# ============================================================
# DIAGNOSE  —  negation + fuzzy + TF-IDF scoring
# ============================================================
# Map age number → DB age_group label
# Duration → severity upgrade map
_DURATION_UPGRADE = {
    "Just started": 0,
    "1-2 days":     0,
    "3-5 days":     1,   # MILD  → MODERATE
    "1 week":       1,   # MILD  → MODERATE
    "2 weeks":      2,   # MILD  → CRITICAL, MODERATE → CRITICAL
    "1 month":      2,
    "3+ months":    2,
}
_SEV_LADDER = ["MILD", "MODERATE", "CRITICAL"]

def upgrade_severity(severity: str, duration: str) -> tuple[str, bool]:
    """Return (new_severity, was_upgraded) based on symptom duration."""
    steps   = _DURATION_UPGRADE.get(duration, 0)
    current = _SEV_LADDER.index(severity.upper()) if severity.upper() in _SEV_LADDER else 1
    new_idx = min(current + steps, len(_SEV_LADDER) - 1)
    new_sev = _SEV_LADDER[new_idx]
    return new_sev, (new_sev != severity.upper())

def _age_to_group(age: int) -> str:
    if age <= 0:       return ""
    if age <= 4:       return "child"
    if age <= 12:      return "child"
    if age <= 17:      return "teenager"
    if age <= 30:      return "young adult"
    if age <= 50:      return "adult"
    if age <= 65:      return "middle-aged"
    return "elderly"

def diagnose_symptoms(user_symptoms: str, hindi_map: dict,
                      user_age: int = 0, user_gender: str = "") -> dict | None:
    normalized   = normalize_symptoms(user_symptoms, hindi_map)
    cleaned      = remove_negated_symptoms(normalized)
    input_words  = [w for w in cleaned.replace(",", " ").split() if len(w) > 2]
    if not input_words:
        return None

    disease_rows = load_disease_rows()   # comes from cache — instant
    idf_map      = build_idf_map()       # comes from cache — instant

    age_group    = _age_to_group(user_age).lower()
    gender_lower = user_gender.lower()

    best_match, top_score = None, 0.0
    for dis, cat, icd, sev, syms, med, consult, spec, row_age, row_gen in disease_rows:
        raw_score = 0.0
        matched   = 0
        for w in input_words:
            if fuzzy_match(w, syms):
                idf = idf_map.get(w, 1.0)
                raw_score += idf
                matched   += 1

        if raw_score == 0:
            continue

        # ── Coverage penalty ────────────────────────────────────
        # A disease with a long symptom list that only partially overlaps
        # with the input should NOT score the same as a disease whose
        # (short) symptom list is almost fully covered by the input.
        # Without this, severe/rare diseases (e.g. HIV/AIDS, which lists
        # "fever, cough" among many other symptoms) can tie with common
        # conditions (e.g. Common Cold) on a shared symptom or two.
        total_disease_symptoms = len(set(syms.split()))
        coverage   = matched / max(total_disease_symptoms, 1)
        input_cov  = matched / max(len(set(input_words)), 1)
        score = raw_score * (0.4 + 0.3 * coverage + 0.3 * input_cov)

        # Boost rows that match user's age group
        if age_group and age_group in row_age.lower():
            score *= 1.3

        # Boost rows that match user's gender
        if gender_lower and gender_lower != "unspecified" and gender_lower in row_gen.lower():
            score *= 1.2

        if score > top_score:
            top_score  = score
            best_match = dict(disease=dis, category=cat, icd_code=icd,
                              severity=sev, medicine=med, consultation=consult,
                              specialist=spec, score=matched)

    return best_match if top_score > 0 else None

# ============================================================
# AUTH  (writes go through write_db)
# ============================================================
def create_user(username: str, password: str, email: str = "",
                age: int = 0, gender: str = "") -> bool:
    try:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        # Safe migrations for new columns
        for sql in [
            "ALTER TABLE users ADD COLUMN email  TEXT DEFAULT ''",
            "ALTER TABLE users ADD COLUMN age    INTEGER DEFAULT 0",
            "ALTER TABLE users ADD COLUMN gender TEXT DEFAULT ''",
        ]:
            try:
                write_db(sql, ())
            except Exception:
                pass
        write_db(
            "INSERT INTO users (username, password, email, age, gender) VALUES (?, ?, ?, ?, ?)",
            (username, hashed, email.strip().lower(), age, gender)
        )
        return True
    except sqlite3.IntegrityError:
        st.error("❌ That username is already taken — please choose another.")
        return False
    except Exception as e:
        st.error(f"❌ Signup failed: {e}")
        return False

def login_user(username: str, password: str):
    row = get_read_conn().execute(
        "SELECT * FROM users WHERE username=?", (username,)
    ).fetchone()
    if not row:
        return None
    try:
        if bcrypt.checkpw(password.encode(), row[2].encode()):
            return row
        return None
    except ValueError as e:
        st.error(
            "❌ This account's password isn't compatible with the current login "
            "system (likely an old plain-text password). Please re-register or "
            "run the password migration script."
        )
        return None

# ============================================================
# HISTORY
# ============================================================
def save_history(username, symptoms, disease, medicine, severity, specialist):
    write_db(
        "INSERT INTO patient_history (username, symptoms, disease, medicine, severity, specialist) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (username, symptoms, disease, medicine, severity, specialist),
    )

def get_history(username: str) -> list:
    with sqlite3.connect("medical.db") as conn:
        return conn.execute(
            "SELECT symptoms, disease, medicine, severity, specialist "
            "FROM patient_history WHERE username=? ORDER BY id DESC",
            (username,),
        ).fetchall()

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================
for key, default in [
    ("logged_in",      False),
    ("username",       ""),
    ("user_email",     ""),
    ("user_age",       0),
    ("user_gender",    ""),
    ("chat_messages",  []),
    ("last_diagnosis", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

hindi_map = load_hindi_map()

# ============================================================
# TITLE
# ============================================================
st.title("🩺 MedAssist AI")
st.subheader("AI-Powered Healthcare Assistant")
st.caption("81 diseases · 11 categories · 5,000 variations · Hindi + English symptom support")

# ============================================================
# ICD BADGE HELPER
# ============================================================
def icd_badge(icd_code: str) -> str:
    """Return styled HTML badge + WHO link for an ICD-10 code."""
    if not icd_code or icd_code == "N/A":
        return "—"
    url = f"https://icd.who.int/browse10/2019/en#/{icd_code}"
    return (
        f'<a href="{url}" target="_blank" style="'
        'background:#1a73e8;color:white;padding:3px 10px;'
        'border-radius:12px;font-size:13px;font-weight:bold;'
        'text-decoration:none;font-family:monospace;letter-spacing:1px;">'
        f'🏥 {icd_code}'
        '</a>'
        '<span style="font-size:11px;color:#888;margin-left:8px;">'
        '(WHO ICD-10 — click to view)</span>'
    )

# ============================================================
# LOGIN / SIGNUP
# ============================================================
if not st.session_state.logged_in:
    menu = st.sidebar.selectbox("Menu", ["Login", "Sign Up"])
    if menu == "Sign Up":
        st.subheader("Create an Account")
        new_user     = st.text_input("Choose a username")
        new_password = st.text_input("Choose a password", type="password")
        new_email    = st.text_input("Email (for CRITICAL alerts)", placeholder="you@example.com")
        new_age      = st.number_input("Your age", min_value=1, max_value=120, value=25)
        new_gender   = st.selectbox("Gender", ["unspecified", "male", "female"])
        if st.button("Create Account"):
            if create_user(new_user, new_password, new_email, int(new_age), new_gender):
                st.success("Account created! Please switch to Login.")
            else:
                st.error("That username is already taken.")
    else:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            row = login_user(username, password)
            if row:
                st.session_state.logged_in  = True
                st.session_state.username   = username
                # Load email/age/gender if columns exist; fallback gracefully
                try:
                    st.session_state.user_email  = row[3] if len(row) > 3 else ""
                    st.session_state.user_age    = row[4] if len(row) > 4 else 0
                    st.session_state.user_gender = row[5] if len(row) > 5 else ""
                except Exception:
                    st.session_state.user_email  = ""
                    st.session_state.user_age    = 0
                    st.session_state.user_gender = ""
                st.rerun()
            else:
                st.error("Incorrect username or password.")

# ============================================================
# DASHBOARD
# ============================================================
else:
    st.sidebar.success(f"👤 {st.session_state.username}")
    page = st.sidebar.selectbox(
        "Navigation",
        ["Chat Diagnosis", "Classic Diagnosis", "History", "Analytics", "Symptom Diary", "Medicine Reminders"],
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in      = False
        st.session_state.username       = ""
        st.session_state.chat_messages  = []
        st.session_state.last_diagnosis = None
        st.rerun()

    # ── Sidebar stats (cached 30 s) ────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Database")
    d_count, v_count, h_count = load_sidebar_stats(st.session_state.username)
    st.sidebar.metric("Unique Diseases", d_count)
    st.sidebar.metric("Variations", f"{v_count:,}")
    st.sidebar.metric("Your Records", h_count)

    st.sidebar.markdown("### 🌐 Categories")
    st.sidebar.markdown(
        "Common · Chronic · Mental Health · "
        "Skin & Eye · Emergency · Cancer · "
        "Kidney · Liver · Neuro · Infectious · Tropical"
    )

    SEV_ICON = {"MILD": "🟢", "MODERATE": "🟡", "CRITICAL": "🔴"}

    CAT_EMOJI = {
        "common": "🤧", "chronic": "💊", "mental_health": "🧠",
        "skin_eye": "👁️", "emergency": "🚨", "cancer": "🎗️",
        "kidney": "🫘", "liver": "🫀", "neuro": "⚡",
        "infectious": "🦠", "tropical": "🌴",
    }

    # ============================================================
    # CHAT DIAGNOSIS  (conversational memory)
    # ============================================================
    if page == "Chat Diagnosis":

        st.header("💬 Chat Diagnosis")
        st.info(
            "Describe your symptoms conversationally. "
            "The AI remembers your full session — ask follow-ups or add new symptoms."
        )

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        chat_duration = st.sidebar.select_slider(
            "⏱️ Symptom duration",
            options=["Just started", "1-2 days", "3-5 days",
                     "1 week", "2 weeks", "1 month", "3+ months"],
            value="1-2 days",
            key="chat_duration_slider",
        )

        user_input = st.chat_input("Describe your symptoms or ask a follow-up…")

        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Diagnosis is instant — pure in-memory scan
            result = diagnose_symptoms(user_input, hindi_map,
                                       user_age=st.session_state.user_age,
                                       user_gender=st.session_state.user_gender)

            # ── Duration-based severity upgrade (chat) ────────────
            if result:
                new_sev, was_upgraded = upgrade_severity(result["severity"], chat_duration)
                if was_upgraded:
                    result["severity"] = new_sev

            history_for_llm = [
                (m["role"], m["content"])
                for m in st.session_state.chat_messages[:-1]
            ]

            if result:
                st.session_state.last_diagnosis = result
                save_history(
                    st.session_state.username, user_input,
                    result["disease"], result["medicine"],
                    result["severity"], result["specialist"],
                )
                if result["severity"].upper() == "CRITICAL" and st.session_state.user_email:
                    send_critical_alert(
                        st.session_state.user_email,
                        st.session_state.username,
                        result["disease"], result["medicine"], result["specialist"],
                    )
                    st.warning("🚨 CRITICAL diagnosis — alert email sent to your registered address.")
                diag_block = (
                    f"Disease   : {result['disease']}\n"
                    f"Category  : {result['category']}\n"
                    f"ICD Code  : {result['icd_code']}\n"
                    f"Severity  : {result['severity']}\n"
                    f"Medicines : {result['medicine']}\n"
                    f"Specialist: {result['specialist']}\n"
                    f"Advice    : {result['consultation']}"
                )
                system_msg = (
                    "You are MedAssist AI, a professional and empathetic medical assistant "
                    "with full memory of this conversation session.\n\n"
                    "Rules:\n"
                    "- Respond in clear, simple English.\n"
                    "- Use the diagnosis data to explain the condition naturally.\n"
                    "- Cover: condition, severity, medicines, specialist, key advice.\n"
                    "- If severity is CRITICAL, urge emergency care immediately.\n"
                    "- Be conversational — reference earlier messages when relevant.\n"
                    "- Keep replies under 180 words.\n"
                    "- End with: ⚠️ Disclaimer: This is not a formal medical diagnosis. "
                    "Please consult a licensed doctor.\n\n"
                    f"Latest diagnosis:\n{diag_block}"
                )
            elif st.session_state.last_diagnosis:
                prev = st.session_state.last_diagnosis
                system_msg = (
                    "You are MedAssist AI with full memory of this conversation.\n\n"
                    f"Most recent diagnosis:\n"
                    f"Disease: {prev['disease']} | Severity: {prev['severity']}\n"
                    f"Medicines: {prev['medicine']} | Specialist: {prev['specialist']}\n\n"
                    "Answer the follow-up using this context. Be conversational, under 180 words.\n"
                    "End with: ⚠️ Disclaimer: This is not a formal medical diagnosis. "
                    "Please consult a licensed doctor."
                )
            else:
                system_msg = (
                    "You are MedAssist AI. No condition matched the patient's message.\n"
                    "Warmly ask for more symptom details (e.g. fever, headache, cough). "
                    "Keep your response under 100 words."
                )

            messages_for_ai = [("system", system_msg)] + history_for_llm + [("human", user_input)]
            with st.spinner("Thinking…"):
                ai_reply = ai_brain.invoke(messages_for_ai).content

            st.session_state.chat_messages.append({"role": "assistant", "content": ai_reply})
            with st.chat_message("assistant"):
                st.markdown(ai_reply)

            if result:
                sev_icon = SEV_ICON.get(result["severity"], "⚪")
                with st.expander("📋 Structured diagnosis card"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Condition:** {result['disease']}")
                        st.markdown("**ICD Code:**"); st.markdown(icd_badge(result["icd_code"]), unsafe_allow_html=True)
                        st.markdown(f"**Category:** {result['category'].replace('_',' ').title()}")
                    with c2:
                        st.markdown(f"**Severity:** {sev_icon} {result['severity']}")
                        st.markdown(f"**Specialist:** {result['specialist']}")
                        st.markdown(f"**Symptom matches:** {result['score']}")
                    st.markdown("**Medicines:** " + result["medicine"])
                    if result["severity"] == "CRITICAL":
                        st.error("🚨 EMERGENCY — Call 112 or go to the nearest emergency room.")

        if st.session_state.chat_messages:
            if st.button("🗑️ Clear conversation"):
                st.session_state.chat_messages  = []
                st.session_state.last_diagnosis = None
                st.rerun()

    # ============================================================
    # CLASSIC DIAGNOSIS
    # ============================================================
    elif page == "Classic Diagnosis":

        st.header("🔍 AI Medical Diagnosis")
        st.info(
            "Enter symptoms in **English or Hindi**.\n\n"
            "**English:** `fever, chest pain, cough` | "
            "**Hindi:** `bukhaar, sir dard, thakaan`"
        )

        all_symptom_tokens = build_symptom_list()

        st.markdown("**Pick symptoms** (type to search) or describe freely below:")
        selected_symptoms = st.multiselect(
            "Symptom autocomplete",
            options=all_symptom_tokens,
            placeholder="Start typing — e.g. fever, headache…",
            label_visibility="collapsed",
        )

        extra = st.text_area(
            "Additional details (optional)",
            placeholder="Any extra symptoms, duration, or context…",
            height=80,
        )

        # Duration selector
        duration = st.select_slider(
            "⏱️ How long have you had these symptoms?",
            options=["Just started", "1-2 days", "3-5 days",
                     "1 week", "2 weeks", "1 month", "3+ months"],
            value="1-2 days",
        )

        # Merge autocomplete selections + free text into one input
        symptoms = ", ".join(selected_symptoms)
        if extra.strip():
            symptoms = (symptoms + " " + extra).strip()

        if st.button("🩺 Run Diagnosis", use_container_width=True):
            if not symptoms.strip():
                st.warning("Please enter your symptoms first.")
            else:
                with st.spinner("Analysing…"):
                    result = diagnose_symptoms(symptoms, hindi_map,
                                           user_age=st.session_state.user_age,
                                           user_gender=st.session_state.user_gender)

                if result:
                    # ── Duration-based severity upgrade ──────────────
                    new_sev, was_upgraded = upgrade_severity(result["severity"], duration)
                    if was_upgraded:
                        st.warning(
                            f"⏱️ Severity upgraded **{result['severity']} → {new_sev}** "
                            f"because symptoms have lasted **{duration}**. "
                            f"Please consult a doctor sooner."
                        )
                        result["severity"] = new_sev
                    save_history(
                        st.session_state.username, symptoms,
                        result["disease"], result["medicine"],
                        result["severity"], result["specialist"],
                    )
                    if result["severity"].upper() == "CRITICAL" and st.session_state.user_email:
                        send_critical_alert(
                            st.session_state.user_email,
                            st.session_state.username,
                            result["disease"], result["medicine"], result["specialist"],
                        )
                        st.warning("🚨 CRITICAL diagnosis — alert email sent to your registered address.")
                    sev_icon  = SEV_ICON.get(result["severity"], "⚪")
                    cat_emoji = CAT_EMOJI.get(result["category"], "🏥")

                    st.success("✅ Diagnosis complete!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### 🦠 Condition");  st.write(f"**{result['disease']}**")
                        st.markdown("### 🏷️ Category");   st.write(f"{cat_emoji} {result['category'].replace('_',' ').title()}")
                        st.markdown("### 📋 ICD Code")
                        st.markdown(icd_badge(result["icd_code"]), unsafe_allow_html=True)
                    with col2:
                        st.markdown("### ⚠️ Severity");   st.write(f"{sev_icon} **{result['severity']}**")
                        st.markdown("### 👨‍⚕️ Specialist"); st.write(result["specialist"])
                        st.markdown("### 🎯 Match Score"); st.write(f"{result['score']} symptom(s) matched")
                        if st.session_state.get("user_age"):
                            gender = st.session_state.get("user_gender","").title() or "Unspecified"
                            st.markdown("### 👤 Diagnosed For")
                            st.write(f"Age {st.session_state.user_age} · {gender}")

                    st.markdown("### 💊 Recommended Medicines")
                    for med in result["medicine"].split(","):
                        if med.strip(): st.markdown(f"- {med.strip()}")
                    st.markdown("### 📝 Doctor's Advice")
                    st.info(result["consultation"])

                    if result["severity"] == "CRITICAL":
                        st.error("🚨 **EMERGENCY** — Call **112** or go to the nearest ER. Do not drive yourself.")

                    prompt = (
                        f"Patient symptoms: {symptoms}\n"
                        f"Disease: {result['disease']} | Severity: {result['severity']}\n"
                        f"Medicines: {result['medicine']} | Specialist: {result['specialist']}\n"
                        f"Advice: {result['consultation']}\n\n"
                        "Write a short, warm, clear response (3–5 sentences). "
                        "If severity is CRITICAL, urge emergency contact. "
                        "End with: ⚠️ This is not a formal medical diagnosis. Please consult a licensed doctor."
                    )
                    with st.spinner("Generating AI advice…"):
                        st.markdown("### 🤖 AI Health Advice")
                        st.write(ai_brain.invoke(prompt).content)
                    st.warning("⚠️ AI-generated guidance only. Always consult a licensed doctor.")
                else:
                    st.error("No match found. Try more symptoms, or visit a General Physician.")

    # ============================================================
    # HISTORY
    # ============================================================
    elif page == "History":

        st.header("📂 Your Medical History")
        history = get_history(st.session_state.username)

        if not history:
            st.info("No records yet. Run a diagnosis first.")
        else:
            st.write(f"Total records: **{len(history)}**")

            # ── Export Buttons ────────────────────────────────────
            import csv, io
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors

            # CSV
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["#", "Symptoms", "Condition", "Medicines", "Severity", "Specialist"])
            for i, (sym, dis, med, sev, spec) in enumerate(history, 1):
                writer.writerow([i, sym, dis, med, sev, spec])

            # PDF
            def build_history_pdf(username, records) -> bytes:
                buf = io.BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=letter,
                                        leftMargin=40, rightMargin=40,
                                        topMargin=50, bottomMargin=40)
                styles = getSampleStyleSheet()
                story  = []

                story.append(Paragraph(f"MedAssist AI — Medical History", styles["Title"]))
                story.append(Paragraph(f"Patient: {username}", styles["Normal"]))
                story.append(Spacer(1, 12))

                table_data = [["#", "Condition", "Severity", "Medicines", "Specialist", "Symptoms"]]
                SEV_COLORS = {"CRITICAL": colors.HexColor("#FF4B4B"),
                              "MODERATE": colors.HexColor("#FFA500"),
                              "MILD":     colors.HexColor("#21BA45")}
                row_colors = [colors.HexColor("#2C3E50")]  # header

                for i, (sym, dis, med, sev, spec) in enumerate(records, 1):
                    table_data.append([str(i), dis, sev, med, spec, sym])
                    row_colors.append(SEV_COLORS.get(sev.upper(), colors.white))

                col_widths = [25, 100, 60, 120, 90, 135]
                t = Table(table_data, colWidths=col_widths, repeatRows=1)

                style_cmds = [
                    ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#2C3E50")),
                    ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
                    ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
                    ("FONTSIZE",    (0,0), (-1,-1), 8),
                    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
                    ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#CCCCCC")),
                    ("VALIGN",      (0,0), (-1,-1), "TOP"),
                    ("WORDWRAP",    (0,0), (-1,-1), True),
                ]
                # Colour the severity cell per row
                for row_idx, (sym, dis, med, sev, spec) in enumerate(records, 1):
                    c = SEV_COLORS.get(sev.upper(), colors.white)
                    style_cmds.append(("BACKGROUND", (2, row_idx), (2, row_idx), c))
                    style_cmds.append(("TEXTCOLOR",  (2, row_idx), (2, row_idx), colors.white))

                t.setStyle(TableStyle(style_cmds))
                story.append(t)
                story.append(Spacer(1, 16))
                story.append(Paragraph(
                    "Disclaimer: This report is AI-generated and not a formal medical diagnosis. "
                    "Please consult a licensed doctor.",
                    styles["Italic"]
                ))
                doc.build(story)
                return buf.getvalue()

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="⬇️ Download as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"{st.session_state.username}_medical_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col2:
                pdf_bytes = build_history_pdf(st.session_state.username, history)
                st.download_button(
                    label="⬇️ Download as PDF",
                    data=pdf_bytes,
                    file_name=f"{st.session_state.username}_medical_history.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            st.markdown("---")
            st.subheader("🩺 Doctor Visit Prep Report")
            st.caption("AI-generated one-page summary formatted the way doctors read it.")

            if st.button("📋 Generate Doctor Visit Report", use_container_width=True):
                with st.spinner("AI is preparing your report…"):

                    # Build a structured summary for the AI
                    history_text = ""
                    for i, (sym, dis, med, sev, spec) in enumerate(history, 1):
                        history_text += f"Record {i}: {dis} [{sev}] — Symptoms: {sym} — Medicines: {med} — Specialist: {spec}\n"

                    prep_prompt = f"""You are a medical documentation assistant. 
A patient named {st.session_state.username} has the following diagnosis history from MedAssist AI:

{history_text}

Generate a professional Doctor Visit Prep Report with these exact sections:
1. PATIENT SUMMARY — age/gender if known, brief overview
2. RECENT CONDITIONS — list each condition with severity
3. RECURRING SYMPTOMS — identify any symptoms that appear multiple times
4. CURRENT MEDICATIONS — list all unique medicines mentioned
5. RECOMMENDED SPECIALISTS — list unique specialists to consult
6. KEY QUESTIONS TO ASK DOCTOR — 3-5 specific questions based on their history
7. RED FLAGS — any CRITICAL conditions that need urgent attention

Keep it concise, professional, and formatted for a doctor to read in under 2 minutes.
End with: "Generated by MedAssist AI — Not a formal medical document." """

                    resp = ai_brain.invoke([
                        ("system", "You are a professional medical documentation assistant. Be concise and structured."),
                        ("human",  prep_prompt)
                    ])
                    report_text = resp.content

                # Show report in app
                st.markdown("---")
                st.markdown(report_text)
                st.markdown("---")

                # Build downloadable PDF of the report
                def build_prep_pdf(username, report_text) -> bytes:
                    buf    = io.BytesIO()
                    doc    = SimpleDocTemplate(buf, pagesize=letter,
                                              leftMargin=50, rightMargin=50,
                                              topMargin=50, bottomMargin=40)
                    styles = getSampleStyleSheet()
                    story  = []

                    story.append(Paragraph("🩺 Doctor Visit Prep Report", styles["Title"]))
                    story.append(Paragraph(f"Patient: {username}", styles["Normal"]))
                    story.append(Paragraph("Generated by MedAssist AI", styles["Italic"]))
                    story.append(Spacer(1, 16))

                    for line in report_text.splitlines():
                        line = line.strip()
                        if not line:
                            story.append(Spacer(1, 6))
                        elif line[0].isdigit() and "." in line[:3]:
                            story.append(Paragraph(f"<b>{line}</b>", styles["Heading3"]))
                        elif line.startswith("-") or line.startswith("•"):
                            story.append(Paragraph(f"&nbsp;&nbsp;{line}", styles["Normal"]))
                        else:
                            story.append(Paragraph(line, styles["Normal"]))

                    story.append(Spacer(1, 20))
                    story.append(Paragraph(
                        "⚠️ This report is AI-generated and not a formal medical document. "
                        "Please consult a licensed doctor.",
                        styles["Italic"]
                    ))
                    doc.build(story)
                    return buf.getvalue()

                prep_pdf = build_prep_pdf(st.session_state.username, report_text)
                st.download_button(
                    label="⬇️ Download Doctor Prep Report as PDF",
                    data=prep_pdf,
                    file_name=f"{st.session_state.username}_doctor_prep_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            st.markdown("---")

            for idx, (symptoms, disease, medicine, severity, specialist) in enumerate(history, 1):
                sev_icon = SEV_ICON.get(severity, "⚪")
                with st.expander(f"Record {idx} — {disease}  {sev_icon} {severity}"):
                    st.write(f"**Symptoms  :** {symptoms}")
                    st.write(f"**Condition :** {disease}")
                    st.write(f"**Medicines :** {medicine}")
                    st.write(f"**Severity  :** {sev_icon} {severity}")
                    st.write(f"**Specialist:** {specialist}")

    # ============================================================
    # SYMPTOM DIARY
    # ============================================================
    elif page == "Symptom Diary":
        st.header("📅 Symptom Diary")
        st.caption("Log how you feel daily — AI detects patterns over time.")

        from datetime import date, timedelta
        import pandas as pd

        tab1, tab2 = st.tabs(["📝 Add Entry", "📈 View Patterns"])

        with tab1:
            st.subheader("How are you feeling today?")
            entry_date = st.date_input("Date", value=date.today())
            c1, c2, c3 = st.columns(3)
            energy = c1.slider("⚡ Energy", 1, 10, 5)
            pain   = c2.slider("🤕 Pain",   1, 10, 1)
            mood   = c3.slider("😊 Mood",   1, 10, 5)
            syms   = st.text_input("Symptoms today (comma separated)", placeholder="headache, fatigue…")
            notes  = st.text_area("Notes", placeholder="Any extra context…", height=80)

            if st.button("💾 Save Entry", use_container_width=True):
                save_diary_entry(
                    st.session_state.username, str(entry_date),
                    energy, pain, mood, syms, notes
                )
                st.success("Entry saved!")

        with tab2:
            entries = get_diary_entries(st.session_state.username)
            if not entries:
                st.info("No entries yet. Add your first entry!")
            else:
                df = pd.DataFrame(entries, columns=["Date","Energy","Pain","Mood","Symptoms","Notes"])
                df["Date"] = pd.to_datetime(df["Date"])

                # Chart
                import plotly.express as px
                fig = px.line(df.sort_values("Date"), x="Date", y=["Energy","Pain","Mood"],
                              title="Your Health Trends", markers=True,
                              color_discrete_map={"Energy":"#21BA45","Pain":"#FF4B4B","Mood":"#1a73e8"})
                st.plotly_chart(fig, use_container_width=True)

                # AI Pattern Detection
                if st.button("🧠 Detect Patterns with AI", use_container_width=True):
                    with st.spinner("Analysing your diary…"):
                        diary_text = "\n".join(
                            f"{r[0]}: Energy={r[1]}, Pain={r[2]}, Mood={r[3]}, Symptoms={r[4]}"
                            for r in entries[:30]
                        )
                        resp = ai_brain.invoke([
                            ("system", "You are a health pattern analyst. Be concise — max 150 words."),
                            ("human",  f"Analyse this symptom diary and find patterns, trends, or warnings:\n{diary_text}")
                        ])
                        st.info(f"🧠 AI Analysis:\n\n{resp.content}")

                # Raw table
                with st.expander("📋 Raw diary entries"):
                    st.dataframe(df[["Date","Energy","Pain","Mood","Symptoms","Notes"]], use_container_width=True)

    # ============================================================
    # MEDICINE REMINDERS
    # ============================================================
    elif page == "Medicine Reminders":
        st.header("💊 Medicine Reminders")
        st.caption("Set up your medicines and never miss a dose.")

        from datetime import datetime

        tab1, tab2 = st.tabs(["➕ Add Reminder", "📋 My Reminders"])

        with tab1:
            st.subheader("Add a new medicine")

            # Auto-fill from last diagnosis
            last = st.session_state.get("last_diagnosis")
            default_med = last["medicine"].split(",")[0].strip() if last else ""

            medicine = st.text_input("Medicine name", value=default_med)
            dose     = st.text_input("Dose", placeholder="e.g. 500mg, 1 tablet")

            st.markdown("**Reminder times** (leave blank if not needed)")
            c1, c2, c3 = st.columns(3)
            time_1 = c1.text_input("Morning",   placeholder="08:00")
            time_2 = c2.text_input("Afternoon", placeholder="14:00")
            time_3 = c3.text_input("Night",     placeholder="21:00")

            if st.button("💾 Save Reminder", use_container_width=True):
                if medicine.strip():
                    save_reminder(st.session_state.username, medicine.strip(),
                                  dose.strip(), time_1, time_2, time_3)
                    st.success(f"Reminder set for **{medicine}**!")
                else:
                    st.warning("Please enter a medicine name.")

        with tab2:
            reminders = get_reminders(st.session_state.username)
            if not reminders:
                st.info("No reminders yet. Add one above!")
            else:
                now_time = datetime.now().strftime("%H:%M")
                st.caption(f"Current time: **{now_time}**")

                for rid, med, dose, t1, t2, t3, _active in reminders:
                    times = [t for t in [t1, t2, t3] if t]
                    # Check if any reminder time is within next 30 min
                    due_soon = False
                    for t in times:
                        try:
                            rem_min = int(t.split(":")[0])*60 + int(t.split(":")[1])
                            now_min = int(now_time.split(":")[0])*60 + int(now_time.split(":")[1])
                            if 0 <= rem_min - now_min <= 30:
                                due_soon = True
                        except Exception:
                            pass

                    label = f"{'🔔 DUE SOON — ' if due_soon else ''}💊 {med}"
                    with st.expander(label):
                        st.write(f"**Dose:** {dose or '—'}")
                        st.write(f"**Times:** {' · '.join(times) if times else 'Not set'}")
                        if due_soon:
                            st.warning("⏰ This medicine is due within 30 minutes!")
                        if st.button(f"🗑️ Delete", key=f"del_{rid}"):
                            delete_reminder(rid)
                            st.rerun()

    # ============================================================
    # ANALYTICS DASHBOARD
    # ============================================================
    elif page == "Analytics":

        st.header("📊 Analytics Dashboard")

        all_df, user_df = load_analytics_data(st.session_state.username)

        if all_df.empty:
            st.info("No diagnosis records yet. Run some diagnoses first.")
        else:
            scope = st.radio(
                "View data for",
                ["My Records", "All Users (anonymised)"],
                horizontal=True,
            )
            df = user_df if scope == "My Records" else all_df

            if df.empty:
                st.warning("No records for the selected scope.")
            else:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Diagnoses",    len(df))
                k2.metric("Unique Conditions",  df["disease"].nunique())
                k3.metric("Critical Cases",     int((df["severity"] == "CRITICAL").sum()))
                k4.metric("Categories Covered", df["category"].nunique())

                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Top 10 Conditions")
                    top = df["disease"].value_counts().head(10).reset_index()
                    top.columns = ["disease", "count"]
                    fig = px.bar(top, x="count", y="disease", orientation="h",
                                 color="count", color_continuous_scale="Blues",
                                 labels={"count": "Diagnoses", "disease": ""})
                    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                                      margin=dict(l=0,r=0,t=10,b=0), height=340,
                                      yaxis={"categoryorder":"total ascending"})
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Severity Breakdown")
                    sev = df["severity"].value_counts().reset_index()
                    sev.columns = ["severity", "count"]
                    fig = px.pie(sev, names="severity", values="count", hole=0.42,
                                 color="severity",
                                 color_discrete_map={"MILD":"#4CAF50","MODERATE":"#FFC107","CRITICAL":"#F44336"})
                    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=340,
                                      legend=dict(orientation="h", y=-0.1))
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("Diagnoses by Category")
                cat = df["category"].value_counts().reset_index()
                cat.columns = ["category", "count"]
                cat["category"] = cat["category"].str.replace("_", " ").str.title()
                fig = px.bar(cat, x="category", y="count", color="category",
                             color_discrete_sequence=px.colors.qualitative.Safe,
                             labels={"count":"Diagnoses","category":"Category"})
                fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                                  height=300, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                col3, col4 = st.columns(2)

                with col3:
                    st.subheader("Specialist Demand")
                    spec = df["specialist"].value_counts().head(8).reset_index()
                    spec.columns = ["specialist", "count"]
                    fig = px.bar(spec, x="count", y="specialist", orientation="h",
                                 color="count", color_continuous_scale="Teal",
                                 labels={"count":"Referrals","specialist":""})
                    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                                      margin=dict(l=0,r=0,t=10,b=0), height=320,
                                      yaxis={"categoryorder":"total ascending"})
                    st.plotly_chart(fig, use_container_width=True)

                with col4:
                    st.subheader("Severity × Category Heatmap")
                    pivot = (
                        df.groupby(["category","severity"]).size()
                          .reset_index(name="count")
                          .pivot(index="category", columns="severity", values="count")
                          .fillna(0)
                    )
                    for s in ["MILD","MODERATE","CRITICAL"]:
                        if s not in pivot.columns: pivot[s] = 0
                    pivot = pivot[["MILD","MODERATE","CRITICAL"]]
                    pivot.index = pivot.index.str.replace("_"," ").str.title()
                    fig = go.Figure(go.Heatmap(
                        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                        colorscale=[[0,"#E3F2FD"],[0.5,"#42A5F5"],[1,"#0D47A1"]],
                    ))
                    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=320,
                                      xaxis_title="Severity", yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                with st.expander("🔍 Raw data table"):
                    out = df.copy()
                    out["category"] = out["category"].str.replace("_"," ").str.title()
                    st.dataframe(out.rename(columns={
                        "disease":"Condition","severity":"Severity",
                        "specialist":"Specialist","category":"Category",
                    }), use_container_width=True, hide_index=True)