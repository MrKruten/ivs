# Обозначаем, что будем использовать официальный образ Python в качестве базового
FROM python:3.12-slim

# Создаем директорию для приложения
WORKDIR /app

# Копируем файл с зависимостями
COPY ./requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY ./src/ .
COPY ./configs/ ./configs/

# Указываем команду для запуска приложения
CMD ["python", "-u", "main.py"]