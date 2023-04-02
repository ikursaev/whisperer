"""Telegram Whisperer bot."""

from datetime import UTC, datetime, timedelta
from enum import Enum
from io import BytesIO
import logging
import tempfile
import typing as t

from environs import Env
import openai
from pydub import AudioSegment
from telegram import File, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


env = Env(eager=False)
env.read_env()


class Server(Enum):
    TEST = 1
    PROD = 2


bot_api_key = env("TELEGRAM_BOT_API_KEY")
whisper_api_key = env("OPENAI_WHISPER_API_KEY")
allowed_group_ids = env.list("ALLOWED_GROUP_IDS", subcast=int)
test_group_ids = env.list("TEST_GROUP_IDS", subcast=int)
SERVER = env.enum("SERVER", type=Server)

env.seal()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO,
)


def convert_ogg_to_mp3(ogg_file: BytesIO, output_file: t.IO) -> None:
    AudioSegment.from_ogg(ogg_file).export(output_file, format="mp3")


class ASR(t.Protocol):
    def transcribe_speech(self, *args, **kwargs) -> str:
        pass


class RateLimiter(t.Protocol):
    def is_limited(self, *args, **kwargs) -> bool:
        pass


class WhisperASR:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def transcribe_speech(self, output_voice_file: t.IO) -> str:
        return openai.Audio.transcribe("whisper-1", output_voice_file)["text"]


class DummyLimiter:
    def is_limited(self) -> bool:
        return False


class RateLimiterByTime:
    def __init__(self, time_limit: timedelta):
        self.time_limit = time_limit
        self.timestamps: dict[int, datetime] = {}

    def is_limited(self, chat_id: int) -> bool:
        now = datetime.now(tz=UTC)
        last_transcription_time = self.timestamps.get(chat_id)

        if last_transcription_time and now - last_transcription_time < self.time_limit:
            return True

        self.timestamps[chat_id] = now
        return False


class GroupFilter:
    def __init__(self, allowed_group_ids: list[int], test_group_ids: list[int]):
        self.allowed_group_ids = allowed_group_ids
        self.test_group_ids = test_group_ids

    def is_allowed(self, chat_id: int) -> bool:
        return chat_id in self.allowed_group_ids

    def is_test(self, chat_id: int) -> bool:
        return chat_id in self.test_group_ids


class VoiceMessageTranscriber:
    def __init__(self, whisper_asr: ASR):
        self.whisper_asr = whisper_asr

    async def transcribe(self, file: File) -> str:
        with BytesIO() as f:
            await file.download_to_memory(out=f)
            f.seek(0)
            temp_output_voice_file = tempfile.NamedTemporaryFile(suffix=".mp3")
            convert_ogg_to_mp3(f, temp_output_voice_file)
            return self.whisper_asr.transcribe_speech(temp_output_voice_file)


class VoiceMessageHandler:
    def __init__(
        self, voice_message_transcriber: VoiceMessageTranscriber,
        rate_limiter: RateLimiter,
        group_filter: GroupFilter):
        self.voice_message_transcriber = voice_message_transcriber
        self.rate_limiter = rate_limiter
        self.group_filter = group_filter

    async def handle(self, update: Update, context: CallbackContext) -> None:
        group_id = update.effective_chat.id
        if not self.group_filter.is_allowed(group_id) and not self.group_filter.is_test(group_id):
            await update.message.reply_text("This bot only works in the allowed group.")
            return

        if self.rate_limiter.is_limited(group_id):
            await update.message.reply_text(
                "You can only transcribe one message per 10 seconds. Please wait.",
                )
            return

        voice = update.message.voice
        if not voice:
            return

        file = await context.bot.get_file(voice.file_id)
        try:
            transcription = await self.voice_message_transcriber.transcribe(file)
            await update.message.reply_text(transcription)
        except Exception:
            logging.exception("Error transcribing speech")
            await update.message.reply_text(
                "An error occurred while transcribing your message. Please try again.",
            )


class TextMessageHandler:
    def __init__(self, rate_limiter: RateLimiter, group_filter: GroupFilter):
        self.rate_limiter = rate_limiter
        self.group_filter = group_filter
        self.messages: dict[int, list[dict[str, str]]] = {}

    async def handle(self, update: Update, context: CallbackContext) -> None:
        group_id = update.effective_chat.id
        if not self.group_filter.is_allowed(group_id) and not self.group_filter.is_test(group_id):
            await update.message.reply_text("This bot only works in the allowed group.")
            return

        text = update.message.text

        if not text:
            return

        if not text.startswith(update.get_bot().name):
            return

        if self.rate_limiter.is_limited(group_id):
            await update.message.reply_text(
                "You can only transcribe one message per 10 seconds. Please wait.",
            )
            return

        self.messages.setdefault(group_id, []).append({"role": "user", "content": text})
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages[group_id],
                max_tokens=2048,
            )
        except openai.error.InvalidRequestError:
            self.messages[group_id].pop(0)
            logging.exception("Context is too big")
            await update.message.reply_text(
                "An error occurred while getting response from ChatGPT. Please try again.",
            )
        except Exception:
            logging.exception("Error getting ChatGPT response")
            await update.message.reply_text(
                "An error occurred while getting response from ChatGPT. Please try again.",
            )
        else:
            content = response.to_dict()["choices"][0]["message"]["content"]
            self.messages[group_id].append({"role": "assistant", "content": content})
            await update.message.reply_text(content)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Send a voice message in a group and the bot will transcribe it!",
    )


def main() -> None:
    if SERVER == "TEST":
        rate_limiter: RateLimiter = DummyLimiter()
    else:
        rate_limiter = RateLimiterByTime(timedelta(seconds=10))

    group_filter = GroupFilter(allowed_group_ids, test_group_ids)

    whisper_asr = WhisperASR(whisper_api_key)
    voice_message_transcriber = VoiceMessageTranscriber(whisper_asr)

    voice_message_handler = VoiceMessageHandler(
        voice_message_transcriber, rate_limiter, group_filter,
    )
    text_message_handler = TextMessageHandler(rate_limiter, group_filter)

    application = ApplicationBuilder().token(bot_api_key).build()
    application.add_handler(MessageHandler(filters.VOICE, voice_message_handler.handle))
    application.add_handler(MessageHandler(filters.TEXT, text_message_handler.handle))
    application.add_handler(CommandHandler("help", help_command))

    application.run_polling()


if __name__ == "__main__":
    main()
