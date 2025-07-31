# create_admin.py
from database import engine, SessionLocal, Base
from models import User

def create_database_and_admin():
    # This creates the .db file and the users table if they don't exist
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    try:
        # Check if an admin user already exists
        admin_user = db.query(User).filter(User.username == "admin").first()
        if admin_user:
            print("Admin user already exists.")
            return

        # If not, create one
        print("Admin user not found, creating one...")
        admin_password = "admin" # Change this to a more secure default if you want
        
        new_admin = User(username="admin", role="admin")
        new_admin.set_password(admin_password)

        db.add(new_admin)
        db.commit()

        print("Admin user 'admin' created successfully.")
        print(f"Default password: {admin_password}")

    finally:
        db.close()

if __name__ == "__main__":
    create_database_and_admin()