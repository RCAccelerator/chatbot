"""
Authentication module for Chainlit.
"""

from abc import ABC, abstractmethod
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from bcrypt import checkpw
import chainlit as cl

from rca_accelerator_chatbot.config import config


# pylint: disable=too-few-public-methods
class Authentification(ABC):
    """Abstract base class for user authentication."""

    @abstractmethod
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate a user by username and password."""
        raise NotImplementedError


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class DatabaseAuthentification(Authentification):
    """Database-backed authentication implementation."""

    def __init__(self):
        self.database_url = config.auth_database_url
        self.metadata = None
        self.users_table = None
        if not self.database_url:
            raise ValueError("AUTH_DATABASE_URL environment variable " +
                             "is not set.")
        self.connect()

    def connect(self):
        """Connect to the database and set up the session."""
        self.engine = create_engine(self.database_url)
        self.session = sessionmaker(bind=self.engine)

    def authenticate(self, username: str, password: str) -> cl.User | None:
        """
        Authenticate a user by checking the username and password
        against the database.
        Args:
            username: Username of the user
            password: Password of the user
        Returns:
            cl.User: User object if authentication is successful,
                      None otherwise
        """
        auth_ok = False

        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        self.users_table = Table('users', self.metadata,
                                 autoload_with=self.engine)
        auth_session = self.session()
        try:
            user = auth_session.query(self.users_table).filter_by(
                        username=username).first()
            if user and checkpw(password.encode('utf-8'),
                                user.password_hash.encode('utf-8')):
                auth_ok = True
        finally:
            auth_session.close()

        if auth_ok:
            cl.logger.info("User %s authenticated successfully.", username)
            return cl.User(
                identifier=username,
            )
        cl.logger.error("Authentication failed for user %s.", username)
        return None


authentification = DatabaseAuthentification()
