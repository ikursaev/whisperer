from enum import Enum

import typed_settings as ts


class Server(Enum):
    TEST = 1
    PROD = 2


@ts.settings(frozen=True, kw_only=True)
class _Settings:
    bot_api_key: str = ts.secret()
    whisper_api_key: str = ts.secret()
    allowed_group_ids: list[int] = ts.secret()
    test_group_ids: list[int] = ts.secret()
    model: str
    server: Server

settings = ts.load(
    cls=_Settings, appname="whisperer", config_files=["settings.toml", ".secrets.toml"],
)
