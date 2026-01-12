from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
hashed = pwd_context.hash("fokzah-8qogwU-hewfuw")
print(hashed)


