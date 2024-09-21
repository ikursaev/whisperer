"""Telegram Whisperer bot."""

import base64
from datetime import UTC, datetime, timedelta
from io import BytesIO
import logging
import typing as t

import openai as oai
from pydub import AudioSegment
from telegram import constants, Bot, File, Message, Update
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
        self, output_voice_file: None | t.BinaryIO = None
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
    def __init__(self) -> None:
        self.collector: dict[int, list[MessageParam]] = {}
        self.encoding = tiktoken.encoding_for_model(settings.model)

    def add(self, chat_id: int, message: MessageParam) -> None:
        self.collector.setdefault(chat_id, []).append(message)
        self.limit_num_tokens(chat_id)

    def get(self, chat_id: int, index: int = 0) -> MessageParam:
        if not self.collector:
            error = "The collector is empty!"
            raise IndexError(error)
        return self.collector[chat_id][index]

    def list(self, chat_id: int) -> list[MessageParam]:
        if chat_id not in self.collector:
            error = "The collector for this chat is empty!"
            raise ValueError(error)
        return self.collector[chat_id]

    def pop(self, chat_id: int, index: int = 0) -> MessageParam:
        if chat_id not in self.collector:
            error = "The collector for this chat is empty!"
            raise ValueError(error)
        return self.collector[chat_id].pop(index)

    def get_num_tokens(self: t.Self, string: str | None) -> int:
        """Returns the number of tokens in a text string."""
        if string is None:
            return 0
        return len(self.encoding.encode(string))

    def get_total_tokens(self: t.Self, chat_id: int) -> int:
        num_tokens = 0
        for m in self.list(chat_id):
            if isinstance(m["content"], str):
                num_tokens += self.get_num_tokens(m["content"])
            else:
                for content in m["content"]:
                    if content["type"] == "text":
                        num_tokens += self.get_num_tokens(content["text"])
                        break
        return num_tokens

    def limit_num_tokens(self: t.Self, chat_id: int) -> None:
        total_tokens = self.get_total_tokens(chat_id)
        while total_tokens > settings.max_input_tokens:
            earliest_message_contents = self.pop(chat_id, 0)["content"]
            for content in earliest_message_contents:
                if content["type"] == "text":
                    total_tokens -= self.get_num_tokens(content["text"])
                    break


class TextMessageHandler:
    def __init__(self, messages: MessageCollector, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.messages = messages

    def is_message_for_bot(self, bot_name: str, message: Message) -> bool:
        text: str = message.text or message.caption
        is_bot_mentioned = text.startswith(bot_name)

        is_reply_to_bot = False
        is_reply_to_whisperer = False
        if reply := message.reply_to_message:
            is_reply_to_bot = reply.from_user.is_bot
            is_reply_to_whisperer = reply.from_user.username == bot_name[1:]

        return is_bot_mentioned or is_reply_to_bot and is_reply_to_whisperer

    async def encode_image(self: t.Self, image: BytesIO) -> str:
        return base64.b64encode(image.read()).decode("utf-8")

    async def handle(self, update: Update, context: CallbackContext[Bot, str, str, str]) -> None:
        group_id = update.effective_chat.id

        logger.info(f"Chat ID: {group_id}")

        message: Message = update.message

        bot_name = update.get_bot().name

        if not self.is_message_for_bot(bot_name, message):
            return

        if self.rate_limiter.is_limited(chat_id=group_id):
            await message.reply_text(
                "You can only transcribe one message per 10 seconds. Please wait.",
            )
            return

        message_content: list[TextContent | ImageContent] = []

        text = message.text or message.caption
        text = text.removeprefix(bot_name)

        system_default_content = ("""

Additional instructions:

The following tags are currently supported:

<b>bold</b>, <strong>bold</strong>
<i>italic</i>, <em>italic</em>
<u>underline</u>, <ins>underline</ins>
<s>strikethrough</s>, <strike>strikethrough</strike>, <del>strikethrough</del>
<span class="tg-spoiler">spoiler</span>, <tg-spoiler>spoiler</tg-spoiler>
<b>bold <i>italic bold <s>italic bold strikethrough <span class="tg-spoiler">italic bold strikethrough spoiler</span></s> <u>underline italic bold</u></i> bold</b>
<a href="http://www.example.com/">inline URL</a>
<a href="tg://user?id=123456789">inline mention of a user</a>
<tg-emoji emoji-id="5368324170671202286">👍</tg-emoji>
<code>inline fixed-width code</code>
<pre>pre-formatted fixed-width code block</pre>
<pre><code class="language-python">pre-formatted fixed-width code block written in the Python programming language</code></pre>
<blockquote>Block quotation started\nBlock quotation continued\nThe last line of the block quotation</blockquote>
<blockquote expandable>Expandable block quotation started\nExpandable block quotation continued\nExpandable block quotation continued\nHidden by default part of the block quotation started\nExpandable block quotation continued\nThe last line of the block quotation</blockquote>

Please note:

Only use the tags mentioned above.
All <, > and & symbols that are not a part of a tag or an HTML entity must be replaced with the corresponding HTML entities (< with &lt;, > with &gt; and & with &amp;).
All numerical HTML entities are supported.
Use only the following named HTML entities: &lt;, &gt;, &amp; and &quot;.
Use nested pre and code tags, to define programming language for pre entity.
Programming language can't be specified for standalone code tags.
""")
        system_default: MessageParam = {"role": "system", "content": system_default_content}
        self.messages.collector.setdefault(group_id, [system_default])

        if text.strip().startswith("system:"):
            for msg in self.messages.collector[group_id]:
                if msg["role"] == "system":
                    msg["content"] = text.removeprefix("system:") + system_default_content
                    break
            await message.reply_text("New system prompt has been adopted.")
            return

        message_content.append({"type": "text", "text": text})

        if photo := message.photo:
            file_id = photo[-1].file_id
            file = await context.bot.get_file(file_id)
            image = BytesIO(await file.download_as_bytearray())
            base64_image = await self.encode_image(image)
            image_content: ImageContent = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            message_content.append(image_content)

        user_message: MessageParam = {"role": "user", "content": message_content}
        self.messages.add(group_id, user_message)

        try:
            response = await client.chat.completions.create(
                model=settings.model,
                messages=self.messages.list(group_id),
                max_tokens=settings.max_output_tokens,
            )
        except Exception:
            logger.exception("Error getting ChatGPT response")
            await update.message.reply_text(
                "An error occurred while getting response from OpenAI API. Please try again.",
            )
            return
        else:
            content: str = response.choices[0].message.content
            contents: TextContent = {"type": "text", "text": content}
            assistant_message: MessageParam = {"role": "assistant", "content": [contents]}
            self.messages.add(group_id, assistant_message)
            if content is not None:
                await message.reply_text(content, parse_mode=constants.ParseMode.HTML)
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
    message_collector = MessageCollector()
    text_message_handler = TextMessageHandler(message_collector, rate_limiter)

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
