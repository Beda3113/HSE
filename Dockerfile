FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Установка системных зависимостей включая unrar
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    unrar \
    && rm -rf /var/lib/apt/lists/*

# Копирование всех файлов включая архив модели
WORKDIR /app
COPY . .

# Распаковка модели (предполагается, что model_epoch_0.rar находится в корне)
RUN unrar x model_epoch_0.rar && \
    mv model_epoch_0.pth /app/ && \
    rm model_epoch_0.rar

# Установка Python зависимостей
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Создание необходимых директорий
RUN mkdir -p /app/output /app/weights

# Запуск приложения
CMD ["python3", "app.py"]
