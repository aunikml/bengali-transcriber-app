# database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# The database file will be created in the same directory
DATABASE_URL = "sqlite:///./transcriber.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} # Needed for SQLite with Streamlit
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()