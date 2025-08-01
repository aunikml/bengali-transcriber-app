# create_admin.py

import sys
from getpass import getpass
from database import SessionLocal, init_db
from models import User

def create_admin_user():
    """A command-line script to create the first admin user."""

    # Initialize the database and create tables if they don't exist
    print("Initializing database...")
    init_db()
    print("Database initialized.")
    
    db = SessionLocal()

    try:
        # Prompt for admin credentials
        print("\n--- Create Admin User ---")
        username = input("Enter admin username: ")

        # Check if the user already exists
        if db.query(User).filter(User.username == username).first():
            print(f"Error: User '{username}' already exists. Please choose a different username.")
            sys.exit(1)

        password = getpass("Enter admin password: ")
        password_confirm = getpass("Confirm admin password: ")

        if password != password_confirm:
            print("Error: Passwords do not match.")
            sys.exit(1)
        
        if not password:
            print("Error: Password cannot be empty.")
            sys.exit(1)

        # Create the new admin user
        admin_user = User(username=username, role="admin")
        admin_user.set_password(password)

        db.add(admin_user)
        db.commit()

        print(f"\n✅ Admin user '{username}' created successfully!")
        print("You can now run the Streamlit app and log in.")

    finally:
        db.close()

if __name__ == "__main__":
    create_admin_user()