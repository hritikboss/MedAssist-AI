import sqlite3
import json
import os

# ============================================================
# CONNECT DATABASE
# ============================================================
conn   = sqlite3.connect("medical.db")
cursor = conn.cursor()

# ============================================================
# DROP OLD TABLES
# ============================================================
cursor.execute("DROP TABLE IF EXISTS users")
cursor.execute("DROP TABLE IF EXISTS diseases")
cursor.execute("DROP TABLE IF EXISTS disease_variations")
cursor.execute("DROP TABLE IF EXISTS patient_history")
cursor.execute("DROP TABLE IF EXISTS hindi_map")

# ============================================================
# USERS TABLE
# ============================================================
cursor.execute("""
CREATE TABLE users (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    email    TEXT DEFAULT ''
)
""")

# ============================================================
# DISEASES TABLE  (81 unique diseases — master)
# ============================================================
cursor.execute("""
CREATE TABLE diseases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    disease      TEXT UNIQUE,
    category     TEXT,
    icd_code     TEXT,
    severity     TEXT,
    symptoms     TEXT,
    medicine     TEXT,
    specialist   TEXT,
    consultation TEXT
)
""")

# ============================================================
# DISEASE VARIATIONS TABLE  (5000 records)
# ============================================================
cursor.execute("""
CREATE TABLE disease_variations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    disease           TEXT,
    category          TEXT,
    icd_code          TEXT,
    severity          TEXT,
    symptoms_reported TEXT,
    all_symptoms      TEXT,
    medicine          TEXT,
    consultation      TEXT,
    specialist        TEXT,
    age_group         TEXT,
    gender            TEXT,
    symptom_duration  TEXT
)
""")

# ============================================================
# PATIENT HISTORY TABLE
# ============================================================
cursor.execute("""
CREATE TABLE patient_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT,
    symptoms   TEXT,
    disease    TEXT,
    medicine   TEXT,
    severity   TEXT,
    specialist TEXT
)
""")

# ============================================================
# HINDI MAP TABLE  (Hindi → English symptoms)
# ============================================================
cursor.execute("""
CREATE TABLE hindi_map (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    hindi_word  TEXT UNIQUE,
    english_word TEXT
)
""")

# ============================================================
# LOAD medical_knowledge_base_v2.json
# ============================================================
json_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "medical_knowledge_base_v2.json"
)

if not os.path.exists(json_path):
    print("ERROR: medical_knowledge_base_v2.json not found!")
    exit(1)

print("Loading medical_knowledge_base_v2.json ...")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ============================================================
# INSERT 81 UNIQUE DISEASES
# ============================================================
print("Inserting unique diseases ...")

specialist_map = {
    "common":       "General Physician",
    "chronic":      "Internal Medicine Specialist",
    "mental_health":"Psychiatrist",
    "skin_eye":     "Dermatologist / Ophthalmologist",
    "emergency":    "Emergency Medicine / ER Doctor",
    "cancer":       "Oncologist",
    "kidney":       "Nephrologist",
    "liver":        "Hepatologist / Gastroenterologist",
    "neuro":        "Neurologist",
    "infectious":   "Infectious Disease Specialist",
    "tropical":     "General Physician / Infectious Disease",
}

disease_rows = []

for category, diseases in data["diseases"].items():
    specialist = specialist_map.get(category, "General Physician")
    for d in diseases:

        raw_sev = d["severity"].lower()
        if any(w in raw_sev for w in ["life-threat","critical","severe"]):
            severity = "CRITICAL"
        elif any(w in raw_sev for w in ["moderate","chronic","progressive"]):
            severity = "MODERATE"
        else:
            severity = "MILD"

        disease_rows.append((
            d["disease"],
            category,
            d["icd_code"],
            severity,
            ", ".join(d["symptoms"]),
            ", ".join(d["medicines"]),
            specialist,
            d["consultation"]
        ))

cursor.executemany("""
INSERT OR IGNORE INTO diseases
    (disease, category, icd_code, severity,
     symptoms, medicine, specialist, consultation)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", disease_rows)

print(f"  ✓ {len(disease_rows)} unique diseases inserted.")

# ============================================================
# INSERT 5000 VARIATIONS
# ============================================================
print("Inserting 5000 variations ...")

variation_rows = [
    (
        v["disease"], v["category"], v["icd_code"],
        v["severity"], v["symptoms_reported"],
        v["all_symptoms"], v["medicine"],
        v["consultation"], v["specialist"],
        v["age_group"], v["gender"], v["symptom_duration"]
    )
    for v in data["variations"]
]

cursor.executemany("""
INSERT INTO disease_variations
    (disease, category, icd_code, severity,
     symptoms_reported, all_symptoms, medicine,
     consultation, specialist, age_group,
     gender, symptom_duration)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", variation_rows)

print(f"  ✓ {len(variation_rows)} variations inserted.")

# ============================================================
# INSERT HINDI MAP
# ============================================================
print("Inserting Hindi symptom map ...")

hindi_rows = [
    (hindi, english)
    for hindi, english in data["hindi_map"].items()
]

cursor.executemany("""
INSERT OR IGNORE INTO hindi_map (hindi_word, english_word)
VALUES (?, ?)
""", hindi_rows)

print(f"  ✓ {len(hindi_rows)} Hindi mappings inserted.")

# ============================================================
# INDEXES for fast search
# ============================================================
cursor.execute("CREATE INDEX IF NOT EXISTS idx_dv_disease  ON disease_variations(disease)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_dv_category ON disease_variations(category)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_dv_severity ON disease_variations(severity)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_dv_icd      ON disease_variations(icd_code)")

# ============================================================
# SAVE
# ============================================================
conn.commit()
conn.close()

print("\n" + "=" * 55)
print("  medical.db READY!")
print(f"  Unique diseases    : {len(disease_rows)}")
print(f"  Symptom variations : {len(variation_rows)}")
print(f"  Hindi mappings     : {len(hindi_rows)}")
print("  Categories covered :")
from collections import Counter
cats = Counter(r[1] for r in disease_rows)
for cat, count in sorted(cats.items()):
    print(f"    {cat:<20} {count} diseases")
print("=" * 55)
