from aiogram import types
from aiogram.dispatcher import FSMContext, middlewares
from aiogram.dispatcher.handler import CancelHandler

import datetime
from keyboards import keyboard
from database import Database


class CheckPaymentMiddleware(middlewares.BaseMiddleware):
    def __init__(self):
        super().__init__()

    async def __check_payment_status(self, message):
        user = Database.get_user_by_telegram_id(message.from_user.id)

        today = datetime.datetime.now()
        if not user:
            Database.add_user(message.from_user.id, message.from_user.username,
                              message.from_user.first_name, message.from_user.last_name)
            user = Database.get_user_by_telegram_id(message.from_user.id)
        if user.free:
            return True
        is_pay = user.subscription_end > today

        if is_pay:
            return True

        return False

    async def __answer_subscribe_message(self, message: types.Message):

        markup = await keyboard.subscribe_kb()
        message = await message.answer("Пожалуйста, оплатите подписку чтобы пользоваться сервисом.", reply_markup=markup)
        raise CancelHandler()

    async def on_process_message(self, message: types.Message, data: dict):
        # state = await data.get("state").get_state()
        # if state:
        #     return
        # telegram_id = message.from_user.id
        # if message.text in ["/start", "/admin"]:
        #     return
        # if message.text in ["Промокод", "Подписка"]:
        #     return
        # if message.content_type == types.ContentType.SUCCESSFUL_PAYMENT:
        #     return

        # is_pay = await self.__check_payment_status(message)
        # if not is_pay:
        #     await self.__answer_subscribe_message(message)
        return
