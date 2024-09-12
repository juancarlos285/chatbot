from flask import Flask
import logging

logging.basicConfig(level=logging.INFO)


def create_app():
    app = Flask(__name__)

    with app.app_context():
        from .routes import bp as routes_bp

        app.register_blueprint(routes_bp)

    return app  # Originally returned app instance
