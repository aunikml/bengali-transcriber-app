# models.py
from sqlalchemy import Column, Integer, String
from database import Base
import bcrypt

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="user") # e.g., 'user' or 'admin'

    def set_password(self, password):
        """Hashes the password and stores it."""
        pwhash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        self.hashed_password = pwhash.decode('utf-8')

    def check_password(self, password):
        """Checks a password against the stored hash."""
        return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password.encode('utf-8'))