"""
db/relational.py

הקובץ הזה אחראי על:
1) ליצור חיבור ל-DB (SQLite)
2) ליצור Session (חיבור עבודה) כדי לבצע פעולות כמו INSERT/SELECT
3) ליצור את הטבלאות בפועל (init_db)
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# DATABASE_URL:
# sqlite:///trial_recruitment.db  -> יוצר/משתמש בקובץ DB בשם trial_recruitment.db בתיקיית הפרויקט
DATABASE_URL = "sqlite:///trial_recruitment.db"

# engine = "המנוע" שמדבר עם ה-DB
# echo=True -> מציג בטרמינל את פקודות ה-SQL ש-SQLAlchemy מייצרת (מאוד עוזר ללמידה ולדיבוג)
engine = create_engine(DATABASE_URL, echo=True)

# SessionLocal -> מפעל ליצירת Sessions
# Session = "שיחת עבודה" מול ה-DB: מוסיפים נתונים, שואלים נתונים, ואז commit/close
SessionLocal = sessionmaker(bind=engine)


def get_session():
    """
    מחזירה Session פתוח.
    אנחנו נשתמש בזה בקוד שירות (patient_service) כדי לבצע שאילתות והכנסות.
    """
    return SessionLocal()


def init_db():
    """
    יוצרת את הטבלאות בפועל בקובץ ה-DB.
    הטבלאות מוגדרות ב-db/models.py, ו-SQLAlchemy יודעת "ליצור אותן" מתוך ה-Models.
    """
    from db.models import Base  # import מקומי כדי למנוע circular imports
    Base.metadata.create_all(bind=engine)
