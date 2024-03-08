import datetime
from database import Database
from aiogram import types
from aiogram.dispatcher import FSMContext, middlewares


class ActivityUpdaterMiddleware(middlewares.BaseMiddleware):
    def __init__(self):
        super().__init__()

    async def on_process_message(self, message: types.Message, data: dict):
        ...
        # telegram_id = message.from_user.id
        # now = datetime.datetime.now()
        # Database.update_user(telegram_id=telegram_id, last_activity=now)
