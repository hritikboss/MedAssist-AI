import sqlite3
import pandas as pd

# ============================================================
# CONNECT DATABASE
# ============================================================
conn = sqlite3.connect("medical.db")

cursor = conn.cursor()

# ============================================================
# DROP OLD DISEASE TABLE
# ============================================================
cursor.execute("DROP TABLE IF EXISTS diseases")

# ============================================================
# CREATE NEW DISEASE TABLE
# ============================================================
cursor.execute("""
CREATE TABLE diseases (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    symptoms TEXT,

    disease TEXT,

    medicine TEXT,

    severity TEXT,

    specialist TEXT
)
""")

# ============================================================
# LOAD CSV FILE
# ============================================================
df = pd.read_csv(
    "datasets/Diseases_and_Symptoms_dataset.csv"
)

# ============================================================
# GET ALL SYMPTOM COLUMNS
# ============================================================
symptom_columns = list(df.columns[:-1])

# LAST COLUMN = DISEASE
disease_column = df.columns[-1]

# ============================================================
# INSERT DATA
# ============================================================
for index, row in df.iterrows():

    symptoms = []

    # FIND ACTIVE SYMPTOMS
    for symptom in symptom_columns:

        try:

            if row[symptom] == 1:

                clean_symptom = symptom.replace("_", " ")

                symptoms.append(clean_symptom)

        except:
            pass

    # CONVERT LIST TO STRING
    symptoms_text = ",".join(symptoms)

    disease = str(row[disease_column])

    # TEMP VALUES
    medicine = "Consult Doctor"

    severity = "MODERATE"

    specialist = "General Physician"

    # INSERT INTO DATABASE
    cursor.execute("""
    INSERT INTO diseases (

        symptoms,
        disease,
        medicine,
        severity,
        specialist

    )
    VALUES (?, ?, ?, ?, ?)
    """, (

        symptoms_text,
        disease,
        medicine,
        severity,
        specialist
    ))

# ============================================================
# SAVE DATABASE
# ============================================================
conn.commit()

conn.close()

print("Dataset imported successfully!")