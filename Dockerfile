FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов проекта
WORKDIR /app
COPY . .

# Установка Python зависимостей
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Создание необходимых директорий
RUN mkdir -p /app/output /app/weights

# Запуск приложения
CMD ["python3", "app.py"]
