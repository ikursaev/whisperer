"""Telegram Whisperer bot."""

import base64
from collections import defaultdict as dd
from datetime import UTC, datetime, timedelta
from io import BytesIO
import logging
import typing as t

import openai as oai
from pydub import AudioSegment
from telegram import Bot, File, Message, Update, constants
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


class ImageUrl(t.TypedDict):
    url: str


class TextContent(t.TypedDict):
    type: t.Required[t.Literal["text"]]
    text: str


class ImageContent(t.TypedDict):
    type: t.Required[t.Literal["image_url"]]
    image_url: ImageUrl


class MessageParam(t.TypedDict, total=False):
    content: t.Required[list[TextContent | ImageContent] | str]
    """The contents of the user message."""

    role: t.Required[t.Literal["user", "assistant", "system"]]
    """The role of the messages author, in this case `user`."""


logger = logging.getLogger()


MESSAGE_COLLECTORS = {}


class DummyLimiter:
    def is_limited(self: t.Self, **kwargs: t.Unpack[RateLimiterKwargs]) -> bool:
        return False


class RateLimiterByTime:
    def __init__(self, time_limit: timedelta):
        self.time_limit = time_limit
        self.timestamps: dict[int, datetime] = {}

    def is_limited(self: t.Self, chat_id: int | None = None) -> bool:
        if not chat_id:
            logger.exception("Chat id is not specified")
            return True
        now = datetime.now(tz=UTC)
        last_transcription_time = self.timestamps.get(chat_id)

        if last_transcription_time and now - last_transcription_time < self.time_limit:
            return True

        self.timestamps[chat_id] = now
        return False


def convert_ogg_to_mp3(ogg_file: t.BinaryIO) -> None:
    AudioSegment.from_ogg(ogg_file).export(ogg_file, format="mp3")
    ogg_file.name = "file.mp3"


class WhisperASR:
    async def transcribe(
        self,
        output_voice_file: None | t.BinaryIO = None,
    ) -> oai.types.audio.Transcription:
        if not output_voice_file:
            logger.exception("Error transcribing speech. Output file is not specified")
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
    ):
        self.voice_message_transcriber = voice_message_transcriber
        self.rate_limiter = rate_limiter

    async def handle(self, update: Update, context: CallbackContext[Bot, str, str, str]) -> None:
        group_id = update.effective_chat.id

        logger.info(f"Chat ID: {group_id}")

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
            logger.exception("Error transcribing speech")
            await update.message.reply_text(
                "An error occurred while transcribing your message. Please try again.",
            )


class MessageCollector:
    def __init__(self, context: CallbackContext[Bot, str, str, str]) -> None:
        self.encoding = tiktoken.encoding_for_model(settings.model)
        self.message_branches: dict[int, list[MessageParam]] = dd(
            lambda: [{"role": "system", "content": settings.default_system_prompt}],
        )
        self.message_graph: dict[int, int] = {}
        self.context = context

    async def get_origin_message_id(self, message_id: int) -> int:
        prev_message_id = self.message_graph[message_id]
        if prev_message_id == message_id:
            return message_id
        return await self.get_origin_message_id(prev_message_id)

    async def set_system_prompt(self, text: str, branch_id: int) -> None:
        for msg in self.message_branches[branch_id]:
            if msg["role"] == "system":
                msg["content"] = text.removeprefix("system:") + settings.default_system_prompt
                break

    async def encode_image(self: t.Self, image: BytesIO) -> str:
        return base64.b64encode(image.read()).decode("utf-8")

    async def get_branch(self, message: Message) -> list[MessageParam]:
        branch_id = await self.get_origin_message_id(message.message_id)
        return self.message_branches[branch_id]

    async def handle_image(self, message: Message) -> ImageContent:
        file_id = message.photo[-1].file_id
        file = await self.context.bot.get_file(file_id)
        image = BytesIO(await file.download_as_bytearray())
        base64_image = await self.encode_image(image)
        image_content: ImageContent = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }
        return image_content

    async def add(
        self,
        message: Message,
        role: t.Literal["user", "system", "assistant"] = "user",
    ) -> None:
        message_id = message.message_id

        reply_to = message.reply_to_message
        reply_to_id = reply_to.message_id if reply_to else message_id

        self.message_graph[message_id] = reply_to_id

        branch_id = await self.get_origin_message_id(message_id)

        text = (message.text or message.caption or "").strip().removeprefix(self.context.bot.name)

        if text.startswith("system:"):
            await self.set_system_prompt(text, branch_id)
            await message.reply_text("New system prompt has been adopted.")
            return

        message_content: list[TextContent | ImageContent] = [{"type": "text", "text": text}]

        if message.photo:
            image_content = await self.handle_image(message)
            message_content.append(image_content)

        user_message: MessageParam = {"role": role, "content": message_content}

        branch = self.message_branches[branch_id]
        branch.append(user_message)

        self.limit_num_tokens(branch)

    def get_num_tokens(self: t.Self, string: str | None) -> int:
        """Returns the number of tokens in a text string."""
        if not string:
            return 0
        return len(self.encoding.encode(string))

    def get_total_tokens(self: t.Self, branch: list[MessageParam]) -> int:
        num_tokens = 0
        for m in branch:
            if isinstance(m["content"], str):
                num_tokens += self.get_num_tokens(m["content"])
            else:
                for content in m["content"]:
                    if content["type"] == "text":
                        num_tokens += self.get_num_tokens(content["text"])
                        break
        return num_tokens

    def limit_num_tokens(self: t.Self, branch: list[MessageParam]) -> None:
        total_tokens = self.get_total_tokens(branch)
        while total_tokens > settings.max_input_tokens:
            if branch[0]["role"] != "system":
                earliest_message_contents = branch.pop(0)["content"]
            else:
                earliest_message_contents = branch.pop(1)["content"]
            for content in earliest_message_contents:
                if content["type"] == "text":
                    total_tokens -= self.get_num_tokens(content["text"])
                    break


class TextMessageHandler:
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter

    def is_branch_start(self, bot_name: str, message: Message) -> bool:
        text: str = (message.text or message.caption or "").strip()
        return text.startswith(bot_name)

    def is_reply(self, bot_name: str, message: Message) -> bool:
        is_reply_to_bot = False
        is_reply_to_whisperer = False
        if reply := message.reply_to_message:
            is_reply_to_bot = reply.from_user.is_bot
            is_reply_to_whisperer = reply.from_user.username == bot_name[1:]

        return is_reply_to_bot and is_reply_to_whisperer

    async def handle(self, update: Update, context: CallbackContext[Bot, str, str, str]) -> None:
        group_id = update.effective_chat.id

        logger.info(f"Chat ID: {group_id}")

        message: Message = update.message

        bot_name = context.bot.name

        is_branch_start = self.is_branch_start(bot_name, message)
        is_reply = self.is_reply(bot_name, message)

        if not (is_branch_start or is_reply):
            return

        if self.rate_limiter.is_limited(chat_id=group_id):
            await message.reply_text(
                "You can only transcribe one message per 10 seconds. Please wait.",
            )
            return

        message_collector: MessageCollector = MESSAGE_COLLECTORS.setdefault(
            group_id,
            MessageCollector(context),
        )

        await message_collector.add(message)

        messages = await message_collector.get_branch(message)

        try:
            response = await client.chat.completions.create(
                model=settings.model,
                messages=messages,
                max_tokens=settings.max_output_tokens,
            )
        except Exception:
            logger.exception("Error getting ChatGPT response")
            await update.message.reply_text(
                "An error occurred while getting response from OpenAI API. Please try again.",
            )
            return
        else:
            openai_response = response.choices[0].message.content or ""
            if openai_response:
                bot_message = await message.reply_text(
                    openai_response, parse_mode=constants.ParseMode.HTML,
                )
                await message_collector.add(bot_message, role="assistant")
            return


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Send a voice message to the group and the bot will transcribe it!",
    )


def main() -> None:
    if settings.server is Server.TEST:
        rate_limiter: RateLimiter = DummyLimiter()
    else:
        rate_limiter = RateLimiterByTime(timedelta(seconds=10))

    whisper_asr = WhisperASR()
    voice_message_transcriber = VoiceMessageTranscriber(whisper_asr)

    voice_message_handler = VoiceMessageHandler(voice_message_transcriber, rate_limiter)
    text_message_handler = TextMessageHandler(rate_limiter)

    application = ApplicationBuilder().token(settings.bot_api_key).build()
    application.add_handler(
        MessageHandler(
            filters.VOICE
            & filters.Chat(chat_id=settings.allowed_group_ids + settings.test_group_ids),
            voice_message_handler.handle,
        ),
    )
    application.add_handler(
        MessageHandler(
            (filters.TEXT | filters.PHOTO)
            & filters.Chat(chat_id=settings.allowed_group_ids + settings.test_group_ids),
            text_message_handler.handle,
        ),
    )
    application.add_handler(CommandHandler("help", help_command))

    application.run_polling()


if __name__ == "__main__":
    main()
