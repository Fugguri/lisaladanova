import yaml
import json
from dataclasses import dataclass


@dataclass
class ButtonsTexts:
    add_profile: str
    edit_profile: str
       

@dataclass
class MessagesText:
    start:str

@dataclass
class TgBot:
    bot_name :str
    token: str
    db_name : str
    database: str
    host: str
    port: int
    user: str
    password: str

    
@dataclass
class Miscellaneous:
    buttons_texts: ButtonsTexts
    messages: MessagesText
    
@dataclass
class Config:
    tg_bot: TgBot
    misc: Miscellaneous

def load_config(path: str = None, texts_path=None):
    file = open(path, "r") 
    config = json.load(file)
    texts = None

    if texts_path is not None:
        with open(texts_path, "r", encoding="utf-8") as stream:
            try:
                texts = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    return Config(
        tg_bot=TgBot(
            bot_name=config["bot_name"],
            db_name = config["db_name"],
            token=config["TOKEN_API"],
            database=config["database"],
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],

        ),
        misc=Miscellaneous(
            buttons_texts = ButtonsTexts(
                add_profile =texts['keyboard']['add_profile'] ,
                edit_profile = texts['keyboard']['edit_profile']),
            messages=MessagesText(
                    start=texts['messages']['start'],
                ),
            )
    )
