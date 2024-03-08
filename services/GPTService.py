import re
import httpx
import asyncio
import requests

from openai import OpenAI
from openai import BadRequestError
from dotenv import dotenv_values


class GPTService:

    def __init__(self) -> None:
        self.config = dotenv_values(".env")

        proxy = self.config["proxy"]
        api_key = self.config['openAi']
        http_transport = httpx.HTTPTransport(local_address="0.0.0.0")
        http_client = httpx.Client(proxies=proxy, transport=http_transport)

        self.openai = OpenAI(api_key=api_key, http_client=http_client)

        self.promt_url = self.config["PROMT_URL"]
        self.users_message = dict()

        self.update_promt()
        self.token_sum = 0

    def update_promt(self):
        self.PROMT = self.__load_document_text(self.promt_url)
        self.base_message_template = [
            {"role": "system", "content": self.PROMT}]
        self.users_message = dict()

    def __load_document_text(self, url: str) -> str:
        # Extract the document ID from the URL
        match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
        if match_ is None:
            raise ValueError('Invalid Google Docs URL')
        doc_id = match_.group(1)

        # Download the document as plain text
        response = requests.get(
            f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
        response.raise_for_status()
        text = response.text
        return text

    def get_user_user_history(self, user_id):
        messages = self.users_message.get(user_id, None)

        if not messages:
            self.users_message[user_id] = self.base_message_template
            return self.users_message[user_id]
        return messages

    def __append_message_to_history(self, user_id: str | int, role: str = "user", content: str = None):
        history = self.get_user_user_history(user_id)
        history.append({"role": role, "content": content})

    async def create_answer(self, message):
        user_id = message.from_user.id
        try:
            self.__append_message_to_history(
                user_id=user_id,
                role="user",
                content=message.text)
            response = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.get_user_user_history(user_id)
            )
            answer = response.choices[0].message.content

            self.__append_message_to_history(
                user_id=user_id,
                role="assistant",
                content=answer)

            return answer
        except BadRequestError as ex:
            print(ex)
            await asyncio.sleep(20)
            await self.create_answer(message)

        except Exception as ex:
            print(ex, "dsf")
            self.users_message[message.from_user.id] = []
            await message.reply("Не понимаю, давай еще раз")
