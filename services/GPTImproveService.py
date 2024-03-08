# импортируем необходимые библиотеки
from dotenv import dotenv_values
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import os
import re
import logging
import httpx

import requests
import re
import matplotlib.pyplot as plt
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from textwrap import fill
import tiktoken


def load_document_text(url: str) -> str:
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


def duplicate_headers_without_hashes(text):
    """
    Дублирует заголовки в тексте, убирая из дубликатов хэши.

    Например:
    '# Заголовок' превращается в:
    '# Заголовок
    Заголовок'
    """

    # Вспомогательная функция, которая будет вызываться для каждого найденного совпадения в тексте
    def replacer(match):
        # match.group() вернет найденный заголовок с хэшами.
        # затем мы добавляем к нему перенос строки и ту же строку, но без хэшей
        return match.group() + "\n" + match.group().replace("#", "").strip()

    # re.sub ищет в тексте все заголовки, начинающиеся с 1 до 3 хэшей, и заменяет их
    # с помощью функции replacer
    result = re.sub(r'#{1,3} .+', replacer, text)

    return result


def split_text(text):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on)
    fragments = markdown_splitter.split_text(text)

    # Подсчет токенов для каждого фрагмента и построение графика
    fragment_token_counts = [num_tokens_from_string(
        fragment.page_content, "cl100k_base") for fragment in fragments]
    plt.hist(fragment_token_counts, bins=50, alpha=0.5, label='Fragments')
    plt.title('Распределение длин чанков в токенах')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.show()

    return fragments


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Возвращает количество токенов в строке"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def insert_newlines(text: str, max_len: int = 170) -> str:
    """
    Функция разбивает длинный текст на строки определенной максимальной длины.
    """
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) > max_len:
            lines.append(current_line)
            current_line = ""
        current_line += " " + word
    lines.append(current_line)
    return "\n".join(lines)


config = dotenv_values(".env")
proxy_url = config["proxy"]
api_key = config['openAi']

logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)

os.environ["OPENAI_API_KEY"] = api_key
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url
client = OpenAI(api_key=api_key,
                http_client=httpx.Client(
                    proxies=proxy_url,
                    transport=httpx.HTTPTransport(
                        local_address="0.0.0.0"),
                ),)
KNOWLEDGE_BASE_URL = 'https://docs.google.com/document/d/10ncUuHLr3qu1yqpa_vOlDSlBbMBlvboQIgxJ7KueS-Q'
knowledge_database = database = load_document_text(KNOWLEDGE_BASE_URL)
# База знаний, которая будет подаваться в langChain


database[:1000]

database = duplicate_headers_without_hashes(database)

database[:10000]


source_chunks = split_text(database)

# Инициализирум модель эмбеддингов
embeddings = OpenAIEmbeddings()

# Создадим индексную базу из разделенных фрагментов текста
db = FAISS.from_documents(source_chunks, embeddings)

# Инструкция для GPT, которая будет подаваться в system
system = load_document_text(
    'https://docs.google.com/document/d/1bWd9isA57K_vzV9Yf1zfOHrx0HpVYOrI0pjOaza0Dj8')


def answer_index(system, search_query, topic, search_index, verbose=2):
    # Поиск релевантных отрезков из базы знаний по вопросу пользователя
    docs = search_index.similarity_search(search_query, k=3)
    if verbose:
        print('\n ===========================================: ')

    message_content = re.sub(r'\n{2}', ' ', '\n '.join(
        [f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    if verbose:
        print('message_content :\n ======================================== \n', message_content)

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


def summarize_questions(dialog):
    """
    Функция возвращает саммаризированный текст диалога.
    """
    messages = [
        {"role": "system", "content": "Ты - нейро-саммаризатор. Твоя задача - саммаризировать диалог, который тебе пришел. Если пользователь назвал свое имя, обязательно отрази его в саммаризированном диалоге"},
        {"role": "user", "content": "Саммаризируй следующий диалог консультанта и пользователя, тебе запрещено удалять из саммаризации имя пользователя: " +
            " ".join(dialog)}
    ]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",     # используем gpt4 для более точной саммаризации
        messages=messages,
        # Используем более низкую температуру для более определенной суммаризации
        temperature=0,
    )

    return completion.choices[0].message.content


def answer_user_question_dialog(system, db, user_question, question_history):
    """
    Функция возвращает ответ на вопрос пользователя.
    """
    summarized_history = ""
    # Если в истории более одного вопроса, применяем суммаризацию
    if len(question_history) > 0:
        summarized_history = "Вот саммаризированный предыдущий диалог с пользователем: " + \
            summarize_questions([q + ' ' + (a if a else '')
                                for q, a in question_history])

    topic = summarized_history + " Актуальный вопрос пользователя: " + user_question

    # Получаем ответ, используя только user_question для поиска в базе данных
    answer_text = answer_index(system, user_question, topic, db)

    question_history.append(
        (user_question, answer_text if answer_text else ''))
    # Выводим саммаризированный текст, который видит модель
    if summarized_history:
        print('****************************')
        print(summarized_history)
        print('****************************')

    return answer_text


def run_dialog(system_doc_url, knowledge_base_url):
    """
    Функция запускает диалог между пользователем и нейро-консультантом.
    """
    # список кортежей, где каждый кортеж содержит пару вопрос-ответ, для отслеживания истории вопросов и ответов во время сессии диалога.
    question_history = []
    while True:
        user_question = input('Пользователь: ')
        if user_question.lower() == 'stop':
            break
        answer = answer_user_question_dialog(
            system_doc_url, knowledge_base_url, user_question, question_history)
        print('Консультант:', insert_newlines(answer))

    return


run_dialog(system, db)
