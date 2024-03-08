import asyncio
import logging
import sys
from database import Database
from config.config import load_config
from keyboards.keyboards import Keyboards
from handlers.users import register_user_handlers
from handlers.admin import register_admin_handlers
from middlewares import environment, check_payment, activity_updater
from aiogram import Bot, Dispatcher, executor, utils
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.contrib.middlewares.logging import LoggingMiddleware
logger = logging.getLogger(__name__)


async def register_all_middlewares(dp, config, keyboards, db, bot,):
    dp.middleware.setup(check_payment.CheckPaymentMiddleware())
    dp.middleware.setup(activity_updater.ActivityUpdaterMiddleware())
    dp.middleware.setup(environment.EnvironmentMiddleware(
        config=config, db=db, keyboards=keyboards, bot=bot))

    # dp.middleware.setup(LoggingMiddleware())


def register_all_handlers(dp, keyboards, db):
    register_admin_handlers(dp, keyboards)
    register_user_handlers(dp, keyboards)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format=u'%(filename)s:%(lineno)d #%(levelname)-8s [%(asctime)s] - %(name)s - %(message)s',
        filename="logs.log"
    )
    logger.info("Starting bot")
    print("Starting bot")
    config = load_config("config/config.json", "config/texts.yml")

    bot = Bot(token=config.tg_bot.token, parse_mode='HTML')

    db = Database
    dp = Dispatcher(bot, storage=MemoryStorage())
    kbs = Keyboards()

    bot['keyboards'] = kbs
    bot['config'] = config

    await register_all_middlewares(dp, config, kbs, db, bot)
    register_all_handlers(dp, kbs, db)

    # start
    dp.skip_updates = False
    try:
        await dp.start_polling()
    finally:
        await dp.storage.close()
        await dp.storage.wait_closed()
        session = await bot.get_session()
        await session.close()

if __name__ == '__main__':
    try:

        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.error("Bot stopped!")
    except utils.exceptions.TerminatedByOtherGetUpdates:
        sys.exit()
