# Whisperer Bot

A Telegram bot that uses OpenAI Whisper API to automatically transcribe voice messages in Telegram groups

### HOW TO USE

1. Clone the repository

2. Optionally create a venv and activate the venv

3. Run

    ```
    pip install -r requirements.txt
    ```
4. Rename **.secrets.example.toml** file to **.secrets.toml**, add keys and allowed groups. Edit **settings.toml**.
5. Run

    ```
    python app.py
    ```
    OR
    ```
    docker-compose up -d
    ```