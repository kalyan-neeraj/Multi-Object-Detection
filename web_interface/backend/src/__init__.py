from flask import Blueprint

# Initialize the Flask blueprint for the application
bp = Blueprint('main', __name__)

from .routes import *  # Import routes to register them with the blueprint