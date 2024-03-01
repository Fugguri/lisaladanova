FROM python:3.11

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Установим директорию для работы

WORKDIR ./

COPY ./requirements.txt ./

# Устанавливаем зависимости и gunicorn
# RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r ./requirements.txt
RUN pip install mysql-connector-python

# Копируем файлы и билд
COPY ./ ./


CMD python3 main.py
# RUN chmod -R 777 ./