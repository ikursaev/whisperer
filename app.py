"""Telegram Whisperer bot."""

from datetime import UTC, datetime, timedelta
from io import BytesIO
import logging
import time
import typing as t

import openai as oai
from pydub import AudioSegment
from telegram import Bot, File, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
import tiktoken

from config import Server, settings
from protocols import ASR, AudioTranscriber, RateLimiter, RateLimiterKwargs


client = oai.AsyncOpenAI(api_key=settings.whisper_api_key)


class MessageParam(t.TypedDict, total=False):
    content: t.Required[str | None]
    """The contents of the user message."""

    role: t.Required[t.Literal["user", "assistant"]]
    """The role of the messages author, in this case `user`."""


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filename="whisperer.log",
)


class DummyLimiter:
    def is_limited(self: t.Self, **kwargs: t.Unpack[RateLimiterKwargs]) -> bool:
        return False


class RateLimiterByTime:
    def __init__(self: t.Self, time_limit: timedelta):
        self.time_limit = time_limit
        self.timestamps: dict[int, datetime] = {}

    def is_limited(self: t.Self, chat_id: None | int = None) -> bool | None:
        if not chat_id:
            logging.exception("Chat id is not specified")
            return True
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


def convert_ogg_to_mp3(ogg_file: t.BinaryIO) -> None:
    AudioSegment.from_ogg(ogg_file).export(ogg_file, format="mp3")
    ogg_file.name = "file.mp3"


class WhisperASR:
    async def transcribe(self, output_voice_file: None | t.BinaryIO = None):
        if not output_voice_file:
            logging.exception("Error transcribing speech. Output file is not specified")
            return None
        return await client.audio.transcriptions.create(model="whisper-1", file=output_voice_file)


class VoiceMessageTranscriber:
    def __init__(self, whisper_asr: ASR):
        self.whisper_asr = whisper_asr

    async def transcribe(self, file: File) -> t.Awaitable:
        with BytesIO() as temp_file:
            await file.download_to_memory(out=temp_file)
            temp_file.seek(0)
            convert_ogg_to_mp3(temp_file)
            return await self.whisper_asr.transcribe(output_voice_file=temp_file)


class VoiceMessageHandler:
    def __init__(
        self,
        voice_message_transcriber: AudioTranscriber[File],
        rate_limiter: RateLimiter,
        group_filter: GroupFilter,
    ):
        self.voice_message_transcriber = voice_message_transcriber
        self.rate_limiter = rate_limiter
        self.group_filter = group_filter

    async def handle(self, update: Update, context: CallbackContext[Bot, str, str, str]) -> None:
        group_id = update.effective_chat.id
        if not self.group_filter.is_allowed(group_id) and not self.group_filter.is_test(group_id):
            await update.message.reply_text("This bot only works in allowed groups.")
            return

        if self.rate_limiter.is_limited(chat_id=group_id):
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
            await update.message.reply_text(transcription.text)
        except Exception:
            logging.exception("Error transcribing speech")
            await update.message.reply_text(
                "An error occurred while transcribing your message. Please try again.",
            )


class TextMessageHandler:
    def __init__(self, rate_limiter: RateLimiter, group_filter: GroupFilter):
        self.rate_limiter = rate_limiter
        self.group_filter = group_filter
        self.messages: dict[int, list[MessageParam]] = {}
        self.encoding = tiktoken.encoding_for_model(settings.model)

    def get_num_tokens(self: t.Self, string: str | None) -> int:
        """Returns the number of tokens in a text string."""
        if string is None:
            return 0
        return len(self.encoding.encode(string))

    def check_num_tokens(self: t.Self, messages: list[MessageParam]) -> None:
        num_tokens = sum(self.get_num_tokens(c) for m in messages if (c := m["content"]))
        while num_tokens > settings.max_input_tokens:
            earliest_message = messages.pop(0)["content"]
            num_tokens -= self.get_num_tokens(earliest_message)

    async def handle(self, update: Update, context: CallbackContext[Bot, str, str, str]) -> None:
        group_id = update.effective_chat.id
        if not self.group_filter.is_allowed(group_id) and not self.group_filter.is_test(group_id):
            await update.message.reply_text("This bot only works with allowed groups.")
            return

        text = update.message.text

        if not text:
            return

        bot_name = update.get_bot().name
        is_bot_mentioned = text.startswith(bot_name)
        is_reply_to_bot = update.message.reply_to_message.from_user.is_bot
        is_reply_to_whisperer = update.message.reply_to_message.from_user.username == bot_name[1:]
        if not is_bot_mentioned and not (is_reply_to_bot and is_reply_to_whisperer):
            return

        if self.rate_limiter.is_limited(chat_id=group_id):
            await update.message.reply_text(
                "You can only transcribe one message per 10 seconds. Please wait.",
            )
            return

        text = text.removeprefix(bot_name)
        user_message: MessageParam = {"role": "user", "content": text}
        self.messages.setdefault(group_id, []).append(user_message)

        self.check_num_tokens(self.messages[group_id])

        while True:
            try:
                response = await client.chat.completions.create(
                    model=settings.model,
                    messages=self.messages[group_id],
                    max_tokens=settings.max_output_tokens,
                )
            except oai.BadRequestError:
                logging.exception(
                    "Context is too big. Number of messages: %s",
                    len(self.messages[group_id]),
                )
                self.messages[group_id].pop(0)
                time.sleep(0.5)
            except Exception:
                logging.exception("Error getting ChatGPT response")
                await update.message.reply_text(
                    "An error occurred while getting response from ChatGPT. Please try again.",
                )
                return
            else:
                content = response.choices[0].message.content
                assistant_message: MessageParam = {"role": "assistant", "content": content}
                self.messages[group_id].append(assistant_message)
                if content is not None:
                    await update.message.reply_text(content)
                return


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Send a voice message in a group and the bot will transcribe it!",
    )


def main() -> None:
    if settings.server is Server.TEST:
        rate_limiter: RateLimiter = DummyLimiter()
    else:
        rate_limiter = RateLimiterByTime(timedelta(seconds=10))

    group_filter = GroupFilter(settings.allowed_group_ids, settings.test_group_ids)

    whisper_asr = WhisperASR()
    voice_message_transcriber = VoiceMessageTranscriber(whisper_asr)

    voice_message_handler = VoiceMessageHandler(
        voice_message_transcriber,
        rate_limiter,
        group_filter,
    )
    text_message_handler = TextMessageHandler(rate_limiter, group_filter)

    application = ApplicationBuilder().token(settings.bot_api_key).build()
    application.add_handler(MessageHandler(filters.VOICE, voice_message_handler.handle))
    application.add_handler(MessageHandler(filters.TEXT, text_message_handler.handle))
    application.add_handler(CommandHandler("help", help_command))

    application.run_polling()


if __name__ == "__main__":
    main()
