"""
db/seed.py

מטרת הקובץ:
1) ליצור את הטבלאות (init_db)
2) להכניס מטופלים סינתטיים (לא אמיתיים!) כדי שנוכל לבדוק את המערכת

איך מריצים:
python -m db.seed
"""

from db.relational import init_db, get_session
from db.models import Patient


def seed_patients():
    # יוצרים session עבודה
    session = get_session()

    # אם כבר יש נתונים, לא נכניס שוב (כדי למנוע כפילויות)
    existing = session.query(Patient).count()
    if existing > 0:
        print(f"✅ DB already has {existing} patients. Skipping seed.")
        session.close()
        return

    # מטופלים סינתטיים לדוגמה
    patients = [
        Patient(patient_id="P001", age=45, sex="F", primary_condition="CKD", egfr=55),
        Patient(patient_id="P002", age=70, sex="M", primary_condition="Diabetes", egfr=28),
        Patient(patient_id="P003", age=62, sex="F", primary_condition="HTN", egfr=40),
        Patient(patient_id="P004", age=34, sex="M", primary_condition="CKD", egfr=90),
        Patient(patient_id="P005", age=58, sex="F", primary_condition="Diabetes", egfr=32),
        Patient(patient_id="P006", age=80, sex="M", primary_condition="CKD", egfr=18),
    ]

    # מוסיפים את כולם ומבצעים commit
    session.add_all(patients)
    session.commit()
    session.close()

    print("✅ Seed completed: inserted synthetic patients.")


if __name__ == "__main__":
    # 1) יוצרים את הטבלאות בקובץ ה-DB
    init_db()

    # 2) מכניסים נתונים סינתטיים
    seed_patients()
