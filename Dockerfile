FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    wget \
    unrar \
    && rm -rf /var/lib/apt/lists/*

# Копирование основных файлов проекта
WORKDIR /app
COPY . .

# Создание директории для LoRA и загрузка моделей
RUN mkdir -p /app/lora && \
    wget -O /app/lora/buzzer.safetensors "https://civitai.tech/api/download/models/86033?type=Model&format=SafeTensor" && \
    wget -O /app/lora/coffee_style.safetensors "https://civitai.tech/api/download/models/108396?type=Model&format=SafeTensor" && \
    # Добавьте другие LoRA модели по аналогии
    # Распаковка основной модели (если нужно)
    unrar x model_epoch_0.rar && \
    mv model_epoch_0.pth /app/ && \
    rm model_epoch_0.rar

# Установка Python зависимостей
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Создание рабочих директорий
RUN mkdir -p /app/output /app/weights

# Запуск приложения
CMD ["python3", "app.py"]
