from app import create_app
import logging

app = create_app()

if __name__ == "__main__":
    # Enable logging
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True, host="0.0.0.0", port=5100)
