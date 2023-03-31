from environs import Env
import logging
import openai
from pydub import AudioSegment
import tempfile
from io import BytesIO
from datetime import datetime, timedelta
from telegram import Update, File
from telegram.ext import MessageHandler, filters, CallbackContext, ApplicationBuilder, CommandHandler, ContextTypes


# You'll need .env file with env variables
env = Env()
env.read_env()


bot_api_key = env("TELEGRAM_BOT_API_KEY") 
whisper_api_key = env("OPENAI_WHISPER_API_KEY")
allowed_group_chat_ids = env.list("ALLOWED_GROUP_CHAT_IDS", subcast=int)

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def convert_ogg_to_mp3(input: BytesIO, output):
    segment = AudioSegment.from_ogg(input)
    segment.export(output, format="mp3")


class WhisperASR:
    def __init__(self, api_key):
        openai.api_key = api_key

    def transcribe_speech(self, voice_data: bytes) -> str:
        temp_output_voice_file = tempfile.NamedTemporaryFile(suffix='.mp3')
        convert_ogg_to_mp3(voice_data, temp_output_voice_file)
        return openai.Audio.transcribe("whisper-1", temp_output_voice_file)['text']


class RateLimiter:
    def __init__(self, time_limit: timedelta):
        self.time_limit = time_limit
        self.timestamps = {}

    def is_limited(self, chat_id: int) -> bool:
        now = datetime.now()
        last_transcription_time = self.timestamps.get(chat_id)

        if last_transcription_time and now - last_transcription_time < self.time_limit:
            return True

        self.timestamps[chat_id] = now
        return False


class GroupFilter:
    def __init__(self, allowed_group_chat_ids: list[int]):
        self.allowed_group_chat_ids = allowed_group_chat_ids

    def is_allowed(self, chat_id: int) -> bool:
        return chat_id in self.allowed_group_chat_ids


class VoiceMessageTranscriber:
    def __init__(self, whisper_asr: WhisperASR):
        self.whisper_asr = whisper_asr

    async def transcribe(self, file: File) -> str:
        with BytesIO() as f:
            await file.download_to_memory(out=f)
            f.seek(0)
            return self.whisper_asr.transcribe_speech(f)


class VoiceMessageHandler:
    def __init__(self, voice_message_transcriber: VoiceMessageTranscriber, rate_limiter: RateLimiter, group_filter: GroupFilter):
        self.voice_message_transcriber = voice_message_transcriber
        self.rate_limiter = rate_limiter
        self.group_filter = group_filter

    async def handle(self, update: Update, context: CallbackContext):
        chat_id = update.effective_chat.id
        if not self.group_filter.is_allowed(chat_id):
            await update.message.reply_text("This bot only works in the allowed group.")
            return

        if self.rate_limiter.is_limited(chat_id):
            await update.message.reply_text("You can only transcribe one message per 10 seconds. Please wait.")
            return

        voice = update.message.voice
        if not voice:
            return

        file = await context.bot.get_file(voice.file_id)
        try:
            transcription = await self.voice_message_transcriber.transcribe(file)
            await update.message.reply_text(transcription)
        except Exception:
            logging.exception(f"Error transcribing speech")
            await update.message.reply_text("An error occurred while transcribing your message. Please try again.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Send a voice message in a group and the bot will transcribe it!")


def main():
    whisper_asr = WhisperASR(whisper_api_key)
    rate_limiter = RateLimiter(timedelta(seconds=10))
    group_filter = GroupFilter(allowed_group_chat_ids)
    voice_message_transcriber = VoiceMessageTranscriber(whisper_asr)
    voice_message_handler = VoiceMessageHandler(voice_message_transcriber, rate_limiter, group_filter)

    application = ApplicationBuilder().token(bot_api_key).build()
    application.add_handler(MessageHandler(filters.VOICE, voice_message_handler.handle))
    application.add_handler(CommandHandler("help", help_command))

    application.run_polling()


if __name__ == '__main__':
    main()