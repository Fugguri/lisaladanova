from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import dotenv
import pandas as pd
# Создаем соединение с базой данных SQLite
env = dotenv.dotenv_values(".env")
DATABASE_URL = env.get("DATABASE_URL")
print(DATABASE_URL)

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Создаем базовый класс для объявления моделей
Base = declarative_base()

# Определяем модель пользователя


class User(Base):
    __tablename__ = 'Users'

    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, nullable=False)
    username = Column(String(100))
    firstname = Column(String(100))
    lastname = Column(String(100))
    last_activity = Column(DateTime, default=datetime.datetime.utcnow)
    # subscription_end = Column(DateTime, default=datetime.datetime.utcnow)
    role = Column(String(100), default="user")

    def __repr__(self):
        return f"<User(telegram_id={self.telegram_id}, username='{self.username}', fullname='{self.firstname} {self.lastname}'),subscription_end='{self.subscription_end}')>"


# Создаем таблицу в базе данных
Base.metadata.create_all(engine)


class UserManager:
    def __init__(self):
        self.session = Session()
        self.session.autoflush = True

    def add_user(self, telegram_id: int, username: str = None, firstname: str = None, lastname: str = None):
        with Session.begin() as session:
            new_user = User(telegram_id=telegram_id, username=username,
                            firstname=firstname, lastname=lastname)
            session.add(new_user)
            session.commit()

    def get_all_users(self):
        with Session() as session:
            all_users = session.query(User).all()
            return all_users

    def get_all_users_for_statistic(self):
        with Session() as session:
            all_users = session.query(User).all()
            return pd.read_sql_table("Users", con=session.bind)
            return all_users

    def get_user_by_telegram_id(self, telegram_id: int):
        with Session() as session:
            user = session.query(User).filter_by(
                telegram_id=telegram_id).first()
        return user

    def get_user_by_contract_id(self, contract_id: str):
        with Session() as session:
            user = session.query(User).filter_by(
                contract_id=contract_id).first()
        return user

    def delete_user(self, telegram_id: int):
        with Session.begin() as session:
            user = session.query(User).filter_by(
                telegram_id=telegram_id).first()
            if user:
                session.delete(user)
                session.commit()

    def update_user(self, telegram_id: int,
                    new_username: str = None,
                    new_firstname: str = None,
                    new_lastname: str = None,
                    contract_id: str = None,
                    subscription_end=None,
                    last_activity=None
                    ):
        with Session.begin() as session:
            user = session.query(User).filter_by(
                telegram_id=telegram_id).first()
            if user:
                if new_username:
                    user.username = new_username
                if new_firstname:
                    user.firstname = new_firstname
                if new_lastname:
                    user.lastname = new_lastname
                if contract_id:
                    user.contract_id = contract_id
                if subscription_end:
                    user.subscription_end = subscription_end
                if last_activity:
                    user.last_activity = last_activity
            self.session.commit()
