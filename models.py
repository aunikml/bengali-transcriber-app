# models.py

from database import Base
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

# The User model is updated with a relationship to link to its transcripts.
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default='user')
    
    # This creates the "one-to-many" relationship.
    # One user can have many transcripts.
    transcripts = relationship("Transcript", back_populates="owner")

    def set_password(self, password):
        self.hashed_password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.hashed_password, password)


# This is the new model for storing saved transcriptions in the database.
class Transcript(Base):
    __tablename__ = 'transcripts'
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    original_filename = Column(String)
    
    # The JSON column type is ideal for storing the structured transcript data
    # (the list of dictionaries with timestamps, speakers, and text).
    content = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # This foreign key links each transcript to a specific user.
    owner_id = Column(Integer, ForeignKey('users.id'))
    
    # This creates the "many-to-one" relationship back to the User model.
    owner = relationship("User", back_populates="transcripts")