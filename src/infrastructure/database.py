import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.infrastructure.config import DATABASE_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class LlmSetting(Base):
    __tablename__ = "llm_settings"
    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String, nullable=False)
    model_name = Column(String, nullable=False, default="openai/gpt-latest")

class UserActivityLog(Base):
    __tablename__ = "user_activity_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    action = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class TelemetryAnalysisRecord(Base):
    __tablename__ = "telemetry_analysis_records"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    filename = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    soh = Column(Integer, nullable=False)
    final_rul = Column(String, nullable=False)
    culprit = Column(String, nullable=False)
    recommendation = Column(Text, nullable=False)
    confidence = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()