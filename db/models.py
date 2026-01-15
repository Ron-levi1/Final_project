"""
db/models.py

כאן אנחנו מגדירים את מבנה הטבלאות.
כל class שמוריש מ-Base הוא טבלה.
"""

from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import declarative_base

# Base הוא "הבסיס" שכל הטבלאות יורשות ממנו
Base = declarative_base()


class Patient(Base):
    """
    טבלת patients - בשלב הראשון מינימלית אבל שימושית.

    למה יש גם id וגם patient_id?
    - id: מפתח פנימי של ה-DB (Primary Key). נוח ליחסים עתידיים.
    - patient_id: מזהה "עסקי" (כמו P001) - מה שהמערכת תציג/תשתמש.
    """
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)

    # מזהה מטופל עסקי - ייחודי
    patient_id = Column(String, unique=True, index=True, nullable=False)

    # נתוני בסיס
    age = Column(Integer, nullable=False)
    sex = Column(String, nullable=False)  # "F" / "M"

    # שדה MVP לאבחנה עיקרית (בהמשך נעשה טבלת diagnoses נפרדת)
    primary_condition = Column(String, nullable=True)

    # דוגמה ל-Lab אחד מרכזי (MVP). בהמשך נעשה טבלת labs עם תאריך/יחידות/סוג בדיקה וכו'
    egfr = Column(Float, nullable=True)
