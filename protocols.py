import typing as t


F_contra = t.TypeVar("F", contravariant=True)


class RateLimiterKwargs(t.TypedDict, total=False):
    chat_id: int | None


class ASRKwargs(t.TypedDict):
    output_voice_file: t.BinaryIO


class ASR(t.Protocol):
    async def transcribe(self: t.Self, **kwargs: t.Unpack[ASRKwargs]) -> t.Awaitable:
        ...



class RateLimiter(t.Protocol):
    def is_limited(self: t.Self, **kwargs: t.Unpack[RateLimiterKwargs]) -> bool:
        ...


class AudioTranscriber(t.Protocol[F_contra]):
    async def transcribe(self: t.Self, file: F_contra) -> t.Awaitable:
        ...
