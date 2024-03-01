import datetime
from aiogram import types
from aiogram import Dispatcher
from aiogram.dispatcher.handler import ctx_data
from aiogram.dispatcher import FSMContext

from services import gpt
from database.Database import UserManager
from config.config import Config
from keyboards.keyboards import Keyboards
from .admin import admin


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
    answer = await gpt.create_answer(message)
    await message.answer(answer)


def register_user_handlers(dp: Dispatcher, kb: Keyboards):

    dp.register_message_handler(start, commands=["start"], state="*")
    dp.register_message_handler(create_response, state="*")
