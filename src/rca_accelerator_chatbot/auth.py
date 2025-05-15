"""
Authentication module for Chainlit.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from sqlalchemy import create_engine, MetaData
from sqlalchemy.sql import select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from bcrypt import checkpw

from rca_accelerator_chatbot.config import config


# pylint: disable=too-few-public-methods
class Authentification(ABC):
    """Abstract base class for user authentication."""

    @abstractmethod
    def authenticate(self, username: str, password: str) -> str | None:
        """Authenticate a user by username and password."""
        raise NotImplementedError

    @abstractmethod
    def verify_token(self, token: str) -> str | None:
        """Verify a token and return the associated username if valid."""
        raise NotImplementedError


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class DatabaseAuthentification(Authentification):
    """Database-backed authentication implementation."""

    def __init__(self):
        self.database_url = config.auth_database_url
        if not self.database_url:
            raise ValueError("AUTH_DATABASE_URL environment variable " +
                             "is not set.")
        self.engine = None
        self.session = None
        self.metadata = MetaData()
        self.users_table = None
        self.tokens_table = None
        self.connect()

    def connect(self):
        """Connect to the database and set up the session."""
        self.engine = create_engine(self.database_url)
        self.session = sessionmaker(bind=self.engine)

        # Initialize metadata and tables
        self.metadata.reflect(bind=self.engine)
        if 'users' in self.metadata.tables:
            self.users_table = self.metadata.tables['users']
        if 'tokens' in self.metadata.tables:
            self.tokens_table = self.metadata.tables['tokens']

    def _load_table(self, table_name):
        """
        Load a table from the database metadata.
        Args:
            table_name: Name of the table to load
        Returns:
            Table object if successful, None otherwise
        """
        try:
            self.metadata.reflect(bind=self.engine)
            if table_name in self.metadata.tables:
                return self.metadata.tables[table_name]
            return None
        except SQLAlchemyError:
            return None

    def authenticate(self, username: str, password: str) -> str | None:
        """
        Authenticate a user by checking the username and password
        against the database.
        Args:
            username: Username of the user
            password: Password of the user
        Returns:
            str: Username if authentication is successful, None otherwise
        """
        auth_ok = False

        # Make sure tables are loaded
        if self.users_table is None:
            self.users_table = self._load_table('users')
            if self.users_table is None:
                return None

        auth_session = self.session()
        try:
            user = auth_session.query(self.users_table).filter_by(
                        username=username).first()
            if user and checkpw(password.encode('utf-8'),
                                user.password_hash.encode('utf-8')):
                auth_ok = True
        except SQLAlchemyError:
            # Log the error in a production environment
            auth_ok = False
        finally:
            auth_session.close()

        if auth_ok:
            return username
        return None

    def verify_token(self, token: str) -> str | None:
        """
        Verify if a token is valid and return the associated username.
        Args:
            token: The token to verify
        Returns:
            str: Username if the token is valid, None otherwise
        """
        # Make sure tokens table is loaded
        if self.tokens_table is None:
            self.tokens_table = self._load_table('tokens')
            if self.tokens_table is None:
                return None

        auth_session = self.session()
        try:
            query = select(self.tokens_table).where(
                (self.tokens_table.c.token == token) &
                (self.tokens_table.c.expires_at > datetime.now(timezone.utc))
            )
            result = auth_session.execute(query).fetchone()

            if result:
                return result.username
            return None
        except SQLAlchemyError:
            # Handle database errors gracefully
            return None
        finally:
            auth_session.close()


authentification = DatabaseAuthentification()
