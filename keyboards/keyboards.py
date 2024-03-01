from config.config import Config
from aiogram.utils.callback_data import CallbackData
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, \
    ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton


class Keyboards:
    def __init__(self) -> None:
        self.start_cd = CallbackData("start", "character_id")
        self.admin_cd = CallbackData("mailing", "command")
        self.mailing_cd = CallbackData("admin", "command")
        self.registration_confirm_cd = CallbackData("confirm", "command")

        self.back_cd = CallbackData("back", "role")

    async def subscribe_kb(self):
        kb = ReplyKeyboardMarkup(resize_keyboard=True)
        kb.add(KeyboardButton(text="Промокод"))
        kb.add(KeyboardButton(text="Подписка"))
        # kb.add(InlineKeyboardButton(
        #     text="Подписка на год", callback_data="pay_year"
        # )
        #    )
        # kb.add(InlineKeyboardButton(text="Назад",
        #        callback_data=self.back_cd.new(role="user")))
        return kb

    async def admin_kb(self):
        kb = InlineKeyboardMarkup()

        kb.add(InlineKeyboardButton(text="Рассылка сообщений",
               callback_data=self.admin_cd.new(command="mail")))
        kb.add(InlineKeyboardButton(text="Статистика",
               callback_data=self.admin_cd.new(command="statistic")))
        kb.add(InlineKeyboardButton(text="Обновить промт",
               callback_data=self.admin_cd.new(command="update_promt")))
        kb.add(InlineKeyboardButton(text="Назад",
               callback_data=self.back_cd.new(role='user')))

        return kb

    async def mailing_kb(self, state: str = None, photo: bool = False):
        kb = InlineKeyboardMarkup()
        if state == "wait_mail_text":
            kb.add(InlineKeyboardButton(text="Назад",
                                        callback_data=self.back_cd.new('admin')))
        if state == 'wait_mail_photo':
            kb.add(InlineKeyboardButton(text="Без фото",
                   callback_data=self.mailing_cd.new("no_photo")))
            if photo:
                kb.add(InlineKeyboardButton(text="Отправить с фото",
                                            callback_data=self.mailing_cd.new("start_with_photo")))
        if state == 'confirm':
            kb.add(InlineKeyboardButton(text="Начать рассылку",
                   callback_data=self.mailing_cd.new("start")))
            kb.add(InlineKeyboardButton(text="Редактировать",
                   callback_data=self.admin_cd.new(command="mail")))
        return kb

    async def back_kb(self, role="user"):
        kb = InlineKeyboardMarkup()

        kb.add(InlineKeyboardButton(text="Назад",
               callback_data=self.back_cd.new(role=role)))

        return kb
