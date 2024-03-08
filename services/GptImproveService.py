# импортируем необходимые библиотеки
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import dotenv_values
import os
import re
import logging
import openai
import httpx

from utils import load_document_text, duplicate_headers_without_hashes, split_text, num_tokens_from_string, insert_newlines

logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)


class GptImprove:
    def __init__(self) -> None:

        self.config = dotenv_values(".env")

        proxy_url = self.config["proxy"]
        api_key = self.config['openAi']
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        http_transport = httpx.HTTPTransport(local_address="0.0.0.0")
        http_client = httpx.Client(proxies=proxy_url, transport=http_transport)

        self.client = OpenAI(api_key=api_key, http_client=http_client)
        # База знаний, которая будет подаваться в langChain
        self.create_db()
        self.SYSTEM_BASE_URL = self.config['SYSTEM_BASE_URL']
        self.SUMMARIZE_BASE_URL = self.config['SUMMARIZE_BASE_URL']
        # Инструкция для GPT, которая будет подаваться в system
        self.system = load_document_text(self.SYSTEM_BASE_URL)
        self.summarizer = load_document_text(self.SUMMARIZE_BASE_URL)

        self.question_history = {}

    def create_db(self):
        self.KNOWLEDGE_BASE_URL = self.config['KNOWLEDGE_BASE_URL']
        self.knowledge_database = load_document_text(
            self.KNOWLEDGE_BASE_URL)
        self.knowledge_database[:1000]
        self.knowledge_database = duplicate_headers_without_hashes(
            self.knowledge_database)
        self.knowledge_database[:10000]
        self.source_chunks = split_text(self.knowledge_database)
        # Инициализирум модель эмбеддингов
        self.embeddings = OpenAIEmbeddings()

        # Создадим индексную базу из разделенных фрагментов текста
        self.db = FAISS.from_documents(self.source_chunks, self.embeddings)

    def reload_settings(self):
        self.create_db()
        self.system = load_document_text(self.SYSTEM_BASE_URL)
        self.summarizer = load_document_text(self.SUMMARIZE_BASE_URL)

        self.question_history = {}

    def answer_index(self, system, search_query, topic, search_index, verbose=0):
        # Поиск релевантных отрезков из базы знаний по вопросу пользователя
        docs = self.db.similarity_search(search_query, k=3)
        if verbose:
            print('\n ===========================================: ')

        message_content = re.sub(r'\n{2}', ' ', '\n '.join(
            [f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
        if verbose:
            print(
                'message_content :\n ======================================== \n', message_content)

        client = OpenAI()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Ответь на вопрос пользователя. Документы с информацией для ответа клиенту: {message_content}\n\n{topic}"}
        ]

        if verbose:
            print('\n ===========================================: ')

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            temperature=0.3
        )
        answer = completion.choices[0].message.content
        return answer

    def answer_user_question_dialog(self, system, db, user_question, question_history):
        """
        Функция возвращает ответ на вопрос пользователя.
        """
        summarized_history = ""
        # Если в истории более одного вопроса, применяем суммаризацию
        if len(question_history) > 0:
            summarized_history = "Вот саммаризированный предыдущий диалог с пользователем: " + \
                self.summarize_questions([q + ' ' + (a if a else '')
                                          for q, a in question_history])

        topic = summarized_history + " Актуальный вопрос пользователя: " + user_question

        # Получаем ответ, используя только user_question для поиска в базе данных
        answer_text = self.answer_index(system, user_question, topic, db)

        question_history.append(
            (user_question, answer_text if answer_text else ''))
        # Выводим саммаризированный текст, который видит модель
        if summarized_history:
            print('****************************')
            print(summarized_history)
            print('****************************')

        return answer_text

    def define_conversation_stage(self, dialog):
        """
        Функция возвращает саммаризированный текст диалога.
        """
        messages = [
            {"role": "system", "content": "Представь, что Ты - нейро-анализатор диалогов. Твоя задача - проанализиоровать диалоги и понять какую из 2х стадий сейчас занимает пользователь.1. Общение 2.Скрипт. Стадия скрипт если пользователь отвечает на вопросы, которые задал консультант."},
            {"role": "user", "content": "Саммаризируй следующий диалог консультанта и пользователя" +
                " ".join(dialog)}
        ]

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",     # используем gpt4 для более точной саммаризации
            messages=messages,
            # Используем более низкую температуру для более определенной суммаризации
            temperature=0,
        )

        return completion.choices[0].message.content

    def summarize_questions(self, dialog):
        """
        Функция возвращает саммаризированный текст диалога.
        """
        messages = [
            {"role": "system", "content": self.summarizer},
            {"role": "user", "content": "Саммаризируй следующий диалог консультанта и пользователя, тебе запрещено удалять из саммаризации имя пользователя: " +
                " ".join(dialog)}
        ]

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",     # используем gpt4 для более точной саммаризации
            messages=messages,
            # Используем более низкую температуру для более определенной суммаризации
            temperature=0,
        )
        print('****************************')
        print(completion.choices[0].message.content)
        print('****************************')
        return completion.choices[0].message.content

    def answer_user_question_dialog(self, system, db, user_question, question_history, user_id):
        """
        Функция возвращает ответ на вопрос пользователя.
        """
        summarized_history = ""
        # Если в истории более одного вопроса, применяем суммаризацию
        if len(question_history) > 0:
            summarized_history = "Вот саммаризированный предыдущий диалог с пользователем: " + \
                self.summarize_questions([q + ' ' + (a if a else '')
                                          for q, a in self.question_history.get(user_id)])
            print('****************************')
            stage = self.define_conversation_stage([q + ' ' + (a if a else '')
                                                   for q, a in self.question_history.get(user_id)])
            if "скрипт" in stage:
                print("СКРИПТ")
            if "общение" in stage:
                print("ОБЩЕНИЕ")
            print('****************************')
        topic = summarized_history + " Актуальный вопрос пользователя: " + user_question

        # Получаем ответ, используя только user_question для поиска в базе данных
        answer_text = self.answer_index(system, user_question, topic, db)
        self.question_history.get(user_id).append(
            (user_question, answer_text if answer_text else ''))
        # Выводим саммаризированный текст, который видит модель
        if summarized_history:
            print('****************************')
            print(summarized_history)
            print('****************************')

        return answer_text

    def run_dialog(self, system_doc_url, knowledge_base_url):
        """
        Функция запускает диалог между пользователем и нейро-консультантом.
        """
        question_history = []
        # список кортежей, где каждый кортеж содержит пару вопрос-ответ, для отслеживания истории вопросов и ответов во время сессии диалога.
        while True:
            user_question = input('Пользователь: ')
            if user_question.lower() == 'stop':
                break
            answer = self.answer_user_question_dialog(
                system_doc_url, knowledge_base_url, user_question, question_history)
            consultant = insert_newlines(answer)
            # print('Консультант:', consultant)

        return consultant

    async def create_answer(self, user_id, message):
        user_question = message
        # print('Пользователь: ', message)
        question_history = self.question_history.setdefault(user_id, [])
        print(question_history)
        answer = self.answer_user_question_dialog(
            self.system, self.db, user_question, question_history, user_id)
        consultant = insert_newlines(answer)
        print('Консультант:', consultant)

        # self.question_history.update(question_history)

        return consultant


# run_dialog(system, db)
