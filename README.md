# Whisperer Bot

A Telegram bot that uses OpenAI Whisper API to automatically transcribe voice messages in Telegram groups

### HOW TO USE

1. Clone the repository

2. Optionally create a venv and activate the venv

3. Run

    ```
    pip install -r requirements.txt
    ```
4. Create a **.env** file, add keys and allowed groups:

    ```
    TELEGRAM_API_KEY=AAAAAAAAAA:BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
    WHISPER_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    ALLOWED_GROUP_IDS=-100,-1000
    TEST_GROUP_IDS=-1000,-2000
    ```
5. Run

    ```
    python app.py
    ```

OR

1. Build a docker image:
   ```
   docker build . -t whisperer
   ```
2. Run a docker container:
   ```
   docker run --name whisperer  -dti --restart unless-stopped --env-file ./.env whisperer env
   ```