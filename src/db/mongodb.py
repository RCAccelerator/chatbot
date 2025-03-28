"""MongoDB connection and operations for the RCA chatbot."""
import chainlit as cl
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

from config import DB_URL, DEFAULT_DB_NAME, DEFAULT_COLLECTION_NAME


class MongoDBClient:
    """MongoDB client for storing conversation history and feedback."""

    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.available = False
        self.connect()

    def connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(DB_URL, serverSelectionTimeoutMS=5000)
            # Validate connection immediately
            self.client.admin.command('ping')
            self.db = self.client[DEFAULT_DB_NAME]
            self.collection = self.db[DEFAULT_COLLECTION_NAME]
            self.available = True
            cl.logger.info("Successfully connected to MongoDB")
        except (ConnectionFailure, PyMongoError) as exception:
            cl.logger.error("Failed to connect to MongoDB: %s",
                            str(exception))
            self.client = None
            self.db = None
            self.collection = None
            self.available = False

    def save_conversation(self, conversation_data):
        """
        Save conversation data to database.

        Args:
            conversation_data: Dictionary containing all conversation data
        """
        if not self.available or self.collection is None:
            cl.logger.warning("MongoDB is not available - conversation data " +
                              "will not be saved")
            return

        try:
            self.collection.insert_one(conversation_data)
        except PyMongoError as e:
            cl.logger.error("Failed to save conversation data: %s", str(e))

    def update_feedback(self, message_id, feedback_value):
        """Update feedback for a message."""
        if not self.available or self.collection is None:
            cl.logger.warning("MongoDB is not available - feedback will " +
                              "not be saved")
            return False

        try:
            filter_msg = {"message_id": message_id}
            value = {"$set": {"feedback": feedback_value}}
            result = self.collection.update_one(filter_msg, value)
            return result.modified_count > 0
        except PyMongoError as e:
            cl.logger.error("Failed to save feedback: %s", str(e))
            return False


# Create singleton instance
mongodb_client = MongoDBClient()
