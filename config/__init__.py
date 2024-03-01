from .config import load_config

cfg = load_config("config/config.json", "config/texts.yml")

__all__ = [
    'cfg',
]
