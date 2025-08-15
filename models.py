# models.py

from database import Base
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default='user')
    
    # One User can have many Transcripts
    transcripts = relationship("Transcript", back_populates="owner", cascade="all, delete-orphan")

    def set_password(self, password):
        self.hashed_password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.hashed_password, password)

class Transcript(Base):
    __tablename__ = 'transcripts'
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    original_filename = Column(String)
    content = Column(JSON) # Stores the list of speaker segments
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    owner_id = Column(Integer, ForeignKey('users.id'))
    
    owner = relationship("User", back_populates="transcripts")
    
    # One Transcript can have many Tags
    # When a Transcript is deleted, all its Tags are also deleted.
    tags = relationship("Tag", back_populates="transcript", cascade="all, delete-orphan")

class Tag(Base):
    __tablename__ = 'tags'
    id = Column(Integer, primary_key=True, index=True)
    
    # The actual text snippet that was selected by the user (e.g., "next week")
    selected_text = Column(String, index=True)
    
    # The tag applied to the snippet (e.g., "Deadline")
    tag_text = Column(String, index=True)
    
    # The full sentence the snippet came from, for providing context
    context_sentence = Column(String)
    
    # Foreign Key to link this tag to a specific transcript
    transcript_id = Column(Integer, ForeignKey('transcripts.id'))
    
    # Relationship back to the Transcript model
    transcript = relationship("Transcript", back_populates="tags")

class TagDefinition(Base):
    __tablename__ = 'tag_definitions'
    id = Column(Integer, primary_key=True, index=True)
    tag_name = Column(String, unique=True, index=True)
    color = Column(String) # e.g., "#0d6efd"