import datetime
from aiogram import types
from aiogram import Dispatcher
from aiogram.dispatcher.handler import ctx_data
from aiogram.dispatcher import FSMContext

from services import gpt, gpt_v2
from database.Database import UserManager
from config.config import Config
from keyboards.keyboards import Keyboards
from .admin import admin
from utils import speech_to_text


async def start(message: types.Message, state: FSMContext):
    cfg: Config = ctx_data.get()['config']
    kb: Keyboards = ctx_data.get()['keyboards']
    db: UserManager = ctx_data.get()['db']
    try:
        db.add_user(message.from_user.id, message.from_user.username,
                    message.from_user.first_name, message.from_user.last_name)
    except Exception as ex:
        print(ex)
    await message.answer(cfg.misc.messages.start)


async def create_response(message: types.Message, state: FSMContext):
    cfg: Config = ctx_data.get()['config']
    kb: Keyboards = ctx_data.get()['keyboards']

    wait = await message.answer("Набираю сообщение ответ…")

    if message.content_type == types.ContentType.VOICE:
        path = f"voice/{message.voice.file_id}.ogg"
        await message.voice.download(path)
        text = await speech_to_text(path)

    elif message.content_type == types.ContentType.TEXT:
        text = message.text
    message.text = text
    answer = await gpt.create_answer(message)
    answer = await gpt_v2.create_answer(message.from_user.id, text)
    await message.answer(answer)
    await wait.delete()


async def re(message: types.Message, state: FSMContext):
    gpt_v2.reload_settings()
    await message.answer("done")


def register_user_handlers(dp: Dispatcher, kb: Keyboards):

    dp.register_message_handler(start, commands=["start"], state="*")
    dp.register_message_handler(re, commands=["re"], state="*")
    dp.register_message_handler(create_response, content_types=[
                                types.ContentType.TEXT, types.ContentType.VOICE], state="*")
