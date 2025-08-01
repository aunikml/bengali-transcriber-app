# database.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# The database URL points to a local SQLite file named 'sqlite.db'.
DATABASE_URL = "sqlite:///./sqlite.db"

# The engine is the core interface to the database.
# The 'connect_args' is required for SQLite to allow it to be used
# in a multi-threaded context like a web application.
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# A SessionLocal class is a factory for creating new database sessions.
# Each database operation (like a query or a save) will use one of these sessions.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base is a class that our ORM models (like User and Transcript) will inherit from.
Base = declarative_base()


def init_db():
    """
    This function imports all models and creates the database tables if they
    do not already exist.
    """
    # Import the models here to ensure they are registered with the Base metadata.
    from models import User, Transcript # <-- MODIFIED: Added Transcript model
    
    # This command inspects the SQLAlchemy models that inherit from Base
    # and creates the corresponding tables in the database.
    Base.metadata.create_all(bind=engine)