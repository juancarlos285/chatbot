from flask import Flask
import logging

# Configure logging to display info level messages
logging.basicConfig(level=logging.INFO)


def create_app():
    # Create a new Flask application instance
    app = Flask(__name__)

    # Use the application context to register blueprints
    with app.app_context():
        # Import and register the routes blueprint
        from .routes import bp as routes_bp
        app.register_blueprint(routes_bp)

    # Return the Flask application instance
    return app
