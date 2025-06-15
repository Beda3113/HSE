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

# Создание директории для LoRA и загрузка всех моделей
RUN mkdir -p /app/lora && \
    echo "Загрузка LoRA моделей..." && \
    wget -q --show-progress -O /app/lora/buzzer.safetensors "https://civitai.tech/api/download/models/86033?type=Model&format=SafeTensor" && \
    wget -q --show-progress -O /app/lora/coffee_style.safetensors "https://civitai.tech/api/download/models/108396?type=Model&format=SafeTensor" && \
    wget -q --show-progress -O /app/lora/additional_style1.safetensors "https://civitai.tech/api/download/models/40999?type=Model&format=SafeTensor" && \
    wget -q --show-progress -O /app/lora/additional_style2.safetensors "https://civitai.tech/api/download/models/87697?type=Model&format=SafeTensor" && \
    echo "Проверка загруженных файлов..." && \
    ls -lh /app/lora/ && \
    file /app/lora/*.safetensors

# Распаковка основной модели
RUN if [ -f "model_epoch_0.rar" ]; then \
    echo "Распаковка основной модели..." && \
    unrar x model_epoch_0.rar && \
    mv model_epoch_0.pth /app/ && \
    rm model_epoch_0.rar; \
    fi

# Установка Python зависимостей
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Создание рабочих директорий
RUN mkdir -p /app/output /app/weights

# Запуск приложения
CMD ["python3", "app.py"]
