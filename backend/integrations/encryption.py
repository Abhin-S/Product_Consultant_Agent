from cryptography.fernet import Fernet, InvalidToken

from config import settings


fernet = Fernet(settings.FERNET_KEY.encode())


def encrypt_token(token: str) -> str:
    return fernet.encrypt(token.encode()).decode()


def decrypt_token(encrypted: str) -> str:
    try:
        return fernet.decrypt(encrypted.encode()).decode()
    except InvalidToken as exc:
        raise ValueError("Unable to decrypt integration token") from exc