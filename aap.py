import streamlit as st
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from torch import nn
import torchvision.utils as vutils
from PIL import Image as PILImage
import cv2
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from diffusers import StableDiffusionImg2ImgPipeline
import os
import tempfile
import logging
import time
from io import BytesIO
import base64
import threading

# --- Настройки ---
RESULT_DIR = tempfile.mkdtemp()
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)
LORA_PATH = "./lora"  # Папка с LoRA стилями

# --- Логирование ---
logger = logging.getLogger("LogoGenerator")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- FastAPI ---
fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)



import torch
import torch.nn as nn
from torch import optim

# Добавьте это:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ------------------------------------------
# 1. Генератор cGAN
# ------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100, text_embedding_dim=128, output_channels=3):
        super(Generator, self).__init__()
        self.text_embedding_dim = text_embedding_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim + text_embedding_dim, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, text_embedding):
        combined = torch.cat([z, text_embedding], dim=1).view(z.size(0), -1, 1, 1)
        return self.model(combined)


# ------------------------------------------
# 2. Инициализация моделей
# ------------------------------------------
TEXT_TO_INDEX = {}
current_index = 0
text_embedder = nn.Embedding(1000, 128).to(device)
generator = Generator().to(device)

if os.path.exists("model_epoch_0.pth"):
    checkpoint = torch.load("model_epoch_0.pth", map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# ------------------------------------------
# 3. Генерация 5 изображений через cGAN
# ------------------------------------------
def generate_cgan_images(label, count=5):
    global current_index
    if label not in TEXT_TO_INDEX:
        TEXT_TO_INDEX[label] = current_index
        current_index += 1

    images = []
    with torch.no_grad():
        noise = torch.randn(count, 100, device=device)
        text_idx = torch.tensor([TEXT_TO_INDEX[label]] * count, device=device)
        text_emb = text_embedder(text_idx)

        image_tensors = generator(noise, text_emb)
        image_tensors = (image_tensors + 1) / 2  # нормализация [0..1]

        for i in range(count):
            img_tensor = image_tensors[i].unsqueeze(0)
            path = f"{RESULT_DIR}/cgan_{label}_{i}.png"
            vutils.save_image(img_tensor, path)
            images.append(path)
    return images


# --- Апскейл через Real-ESRGAN ---
def init_real_upscaler():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, scale=4)
    weights_path = load_file_from_url(
        url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 
        model_dir='weights', progress=True, file_name='RealESRGAN_x4plus.pth')
    return RealESRGANer(model_path=weights_path, model=model, scale=4, tile=400, half=False, device=device)

def upscale_image(input_path, label):
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        upscaler = init_real_upscaler()
        result, _ = upscaler.enhance(img, outscale=4)
        output_path = input_path.replace(".png", "_upscaled.png")
        cv2.imwrite(output_path, result)
        return output_path
    except Exception as e:
        logger.error(f"Ошибка в upscale_image: {str(e)}")
        return None

# --- Получение списка доступных LoRA стилей ---
def get_available_loras():
    if not os.path.exists(LORA_PATH):
        os.makedirs(LORA_PATH, exist_ok=True)
        return []
    
    loras = []
    for file in os.listdir(LORA_PATH):
        if file.endswith(('.safetensors', '.pt', '.bin')):
            loras.append(file)
    return loras

# --- Стилизация через SD 1.5 ---
@st.cache_resource
def load_diffusion_pipeline(lora_name=None):
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        logger.info(f"Загрузка модели: {model_id}")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)

        if lora_name:
            lora_path = os.path.join(LORA_PATH, lora_name)
            try:
                logger.info(f"Попытка загрузить LoRA веса из: {lora_path}")
                pipe.load_lora_weights(
                    os.path.dirname(lora_path),
                    weight_name=os.path.basename(lora_path)
                )
                logger.info(f"✅ LoRA стиль '{lora_name}' успешно применён")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось загрузить LoRA стиль '{lora_name}': {str(e)}")

        if device == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()

        return pipe
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке пайплайна: {str(e)}", exc_info=True)
        st.error(f"Не удалось загрузить модель: {str(e)}")
        raise

def style_with_diffusion(image_path, label, strength=0.7, steps=30, guidance=13.0, lora_name=None):
    try:
        init_image = PILImage.open(image_path).convert("RGB")
        
        # Минимальный размер 512x512
        if min(init_image.size) < 512:
            init_image = init_image.resize(
                (max(512, init_image.width), max(512, init_image.height)),
                PILImage.LANCZOS
            )

        pipe = load_diffusion_pipeline(lora_name)

        prompt = f"профессиональный логотип: {label}, минимализм, векторная графика"
        negative_prompt = "размытость, артефакты, водяные знаки, низкое качество"

        try:
            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                negative_prompt=negative_prompt
            ).images[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("⚠️ Недостаточно памяти, уменьшаем разрешение...")
            init_image = init_image.resize((512, 512))
            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                negative_prompt=negative_prompt
            ).images[0]

        final_path = f"output/logo_{label}_{int(time.time())}.png"
        result.save(final_path)
        return final_path
    except Exception as e:
        logger.error(f"Ошибка стилизации: {str(e)}")
        return None

# --- Streamlit UI ---
import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from typing import Optional

def run_streamlit():
    # --- Инициализация session_state ---
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    if 'selected_idx' not in st.session_state:
        st.session_state.selected_idx = None
    if 'final_image' not in st.session_state:
        st.session_state.final_image = None
    if 'eda_clicked' not in st.session_state:
        st.session_state.eda_clicked = False
    if 'dataset_for_EDA' not in st.session_state:
        st.session_state.dataset_for_EDA = None
    if 'info_clicked' not in st.session_state:
        st.session_state.info_clicked = False



    st.set_page_config(page_title="Генератор логотипов", layout="wide")
    st.title("🎨 Генератор логотипов")  

    label = st.text_input("Название логотипа:", placeholder="Например: EcoTech")

    if st.button("Сгенерировать варианты") and label:
        with st.spinner("Генерация вариантов..."):
            base_images = generate_cgan_images(label, count=5)
            upscaled_images = [upscale_image(img, label) for img in base_images]
            st.session_state.generated_images = [img for img in upscaled_images if img is not None]
            st.session_state.label = label
            st.session_state.selected_idx = None
            st.session_state.final_image = None

   

    if st.session_state.generated_images:
        st.subheader("Выберите понравившийся вариант:")
        cols = st.columns(5)

        for idx, col in enumerate(cols):
            if idx < len(st.session_state.generated_images):
                image_path = st.session_state.generated_images[idx]
                with col:
                    if st.button(f"Выбрать №{idx + 1}", key=f"select_{idx}"):
                        st.session_state.selected_idx = idx
                    st.image(
                        image_path,
                        caption=f"Шаблон №{idx + 1}",
                        width=150  # <-- Здесь задаём размер
                    )


        if st.session_state.selected_idx is not None:
            selected_img = st.session_state.generated_images[st.session_state.selected_idx]
            st.success(f"Выбран вариант #{st.session_state.selected_idx+1}")

            with st.expander("Настройки стилизации", expanded=True):
                strength = st.slider("Интенсивность стилизации", 0.3, 0.9, 0.7, 0.05)
                steps = st.slider("Количество шагов", 10, 100, 30, 5)
                guidance = st.slider("Контроль стиля", 7.0, 20.0, 13.0, 0.5)
                
                # Выбор LoRA стиля
                available_loras = get_available_loras()
                lora_options = ["Нет"] + available_loras
                selected_lora = st.selectbox("Выберите LoRA стиль", lora_options)

            if st.button("Применить стиль"):
                with st.spinner("Стилизация..."):
                    final_path = style_with_diffusion(
                        selected_img,
                        st.session_state.label,
                        strength,
                        steps,
                        guidance,
                        selected_lora if selected_lora != "Нет" else None
                    )
                    st.session_state.final_image = final_path



            if st.session_state.final_image:
                st.markdown("###  Финальный логотип")
                st.image(
                    st.session_state.final_image,
                    caption=" Ваш финальный логотип",
                    width=512  # <-- здесь задаём нужный размер
                )
                with open(st.session_state.final_image, "rb") as f:
                    st.download_button(
                        "📥 Скачать логотип",
                        data=f,
                        file_name=f"{st.session_state.label}_logo.png",
                        mime="image/png"
                    )

    st.title("ℹ️ Информация о модели")

    themes = [
        "plotly",
        "ggplot2",
        "seaborn",
        "simple_white",
        "presentation",
        "streamlit",
    ]

    def dataset_to_eda(df):
        dataset_transform = pd.DataFrame()
        dataset_transform["epitets_num"] = df["text"].apply(lambda x: len(x.split(",")[1:]))
        dataset_transform["description"] = df["text"].apply(lambda x: ",".join(x.split(",")[1:]))
        dataset_transform["len"] = df["text"].apply(lambda x: len(x))
        dataset_transform["shape"] = df["image"].apply(lambda x: np.array(x).shape)
        dataset_transform["h"] = df["image"].apply(lambda x: np.array(x).shape[0])
        dataset_transform["w"] = df["image"].apply(lambda x: np.array(x).shape[1])
        dataset_transform["rgb"] = df["image"].apply(
            lambda x: "RGB" if len(np.array(x).shape) == 3 else "BW"
        )
        dataset_transform["ratio"] = dataset_transform["shape"].apply(lambda x: x[0] / x[1])
        dataset_transform["pixel"] = dataset_transform["shape"].apply(lambda x: np.prod(x))

        return dataset_transform

    with st.expander("Краткая информация о модели: свернуть/развернуть", expanded=True):
        if st.button("Краткая информация о модели"):
            st.session_state.info_clicked = True

        if st.session_state.info_clicked:
            st.header("Краткая справка")
            st.markdown(
                f"1. cGAN генерирует базовые варианты логотипов из текстового описания\n"
                f"   • Генератор: 4-слойная трансposed CNN (100D шум + 128D текстовый эмбеддинг)\n"
                f"   • Обучение: 15 эпох, метрика FID\n\n"
                f"2. Real-ESRGAN повышает качество изображений (4x апскейл)\n\n"
                f"3. Stable Diffusion 1.5 с LoRA-адаптацией применяет стилизацию:\n"
                f"   • Минимальный размер 512x512\n"
                f"   • Поддержка пользовательских стилей через LoRA\n"
                f"   • Автоматическое уменьшение разрешения при нехватке памяти\n\n"
                f"Ссылка на [репозиторий GitHub](https://github.com/HerrVonBeloff/AI-YP_24-team-42).", 
                unsafe_allow_html=True
            )
            st.header("Процесс обучения cGAN")

            theme_model = st.selectbox(
                "Выберите тему для графиков", sorted(themes), key="theme_selector1"
            )

            try:
                df_loss = pd.read_csv("loss_long.csv")
                df_metric = pd.read_csv("metric.csv")
            except Exception as e:
                st.error("❌ Не удалось загрузить CSV файлы. Проверьте их наличие.")
                df_loss = pd.DataFrame()
                df_metric = pd.DataFrame()

            if not df_loss.empty:
                fig = px.line(
                    df_loss,
                    x="Epoch",
                    y="Loss Value",
                    color="Loss Type",
                    title="Потери",
                    template=theme_model
                )
                fig.update_layout(
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    title=dict(x=0.4)
                )
                fig.update_layout(
                    xaxis_title="Эпохи",
                    yaxis_title="Значения потерь",
                    legend=dict(title="Нейросеть")
                )
                fig.update_traces(name="Дискриминатор", selector=dict(name="D_loss"))
                fig.update_traces(name="Генератор", selector=dict(name="G_loss"))
                st.plotly_chart(fig)

            st.header("Метрика FID")
            if not df_metric.empty:
                fig = px.line(
                    df_metric,
                    x="Epoch",
                    y="FID score",
                    title="Метрика FID",
                    template=theme_model
                )
                fig.update_layout(
                    bargap=0.1,
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    title=dict(x=0.4)
                )
                fig.update_layout(
                    xaxis_title="Эпохи",
                    yaxis_title="Значения метрики"
                )
                st.plotly_chart(fig)

    with st.expander("Загрузка данных: свернуть/развернуть", expanded=True):
        st.header("Загрузка данных")
        example = {
            "image": ["{'bytes': b'\\x89PNG\\n\\x1a\\n'}"],
            "text": ["Simple elegant logo for Concept, love orange ..."]
        }
        example_df = pd.DataFrame(example)
        example_df.index = range(456, 457)
        st.markdown("Требования к датасету")
        st.write(example_df)
        example_dataset_url = "https://drive.google.com/file/d/1BiUi9TOVgIjEggFQHb9d49Dp-z0pgIvI/view?usp=sharing"
        st.markdown(
            "Формат `parquet`. "
            "Изображения представлены в байтовом виде внутри словаря, "
            "текст представлен в виде обычной строки с перечислением эпитетов. "
            f"[🔗 Пример датасета]({example_dataset_url})"
        )

        uploaded_file = st.file_uploader("Загрузите датасет (.parquet)", type=["parquet"])

        if uploaded_file is not None:
            try:
                dataset = pd.read_parquet(uploaded_file)
                dataset["image"] = dataset["image"].apply(
                    lambda x: Image.open(BytesIO(x.get("bytes")))
                )
                st.markdown(
                    f'<span style="color:gray">Количество объектов в датасете: </span>{len(dataset)}',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"❌ Ошибка чтения датасета: {e}")
                logger.error(f"Ошибка чтения датасета: {e}")

    if uploaded_file is not None:
        try:
            dataset = pd.read_parquet(uploaded_file)
            dataset["image"] = dataset["image"].apply(
                lambda x: Image.open(BytesIO(x.get("bytes")))
            )
            st.session_state.dataset = dataset
        except Exception as e:
            st.error(f"❌ Ошибка извлечения изображения: {e}")
            logger.error(f"Ошибка извлечения изображения: {e}")

    with st.expander("Получить случайный элемент датасета: свернуть/развернуть", expanded=True):
        if st.button("Получить случайный элемент датасета"):
            ind = random.randint(0, len(dataset) - 1) if len(dataset) > 0 else 0
            st.session_state.index = ind
            st.session_state.dataset_image = dataset.iloc[ind]["image"]
            st.session_state.dataset_text = dataset.iloc[ind]["text"]

        if "dataset_image" in st.session_state:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(
                    st.session_state.dataset_image,
                    caption=f"Логотип из датасета: индекс {st.session_state.index}",
                )
            st.markdown(
                f'<span style="color:gray">Текстовое описание логотипа: </span>{st.session_state.dataset_text}',
                unsafe_allow_html=True
            )

    with st.expander("EDA: свернуть/развернуть", expanded=True):
        if st.button("EDA"):
            st.session_state.eda_clicked = True
            try:
                dataset_for_eda = dataset_to_eda(dataset)
                st.session_state.dataset_for_EDA = dataset_for_eda
            except Exception as e:
                st.error(f"❌ Ошибка форматирования датасета: {e}")
                logger.error(f"Ошибка форматирования датасета: {e}")

        if st.session_state.eda_clicked:
            if st.session_state.dataset_for_EDA is not None:
                data = st.session_state.dataset_for_EDA
                st.title("Анализ текстовых данных")
                st.subheader("Облако слов")

                if "word_cloud" not in st.session_state:
                    wc = WordCloud(background_color="black", width=1000, height=500)
                    words = data["description"].explode().values
                    wc.generate(" ".join(words))
                    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                    ax.axis("off")
                    ax.set_title("Облако слов для описаний логотипов", fontsize=30)
                    ax.imshow(wc, alpha=0.98)
                    st.session_state.word_cloud = fig

                st.pyplot(st.session_state.word_cloud)

                theme = st.selectbox(
                    "Выберите тему для графиков", sorted(themes), key="theme_selector"
                )

                st.subheader("Гистограмма распределения длин описаний")
                bins = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_description"
                )
                bin_edges = np.linspace(data["len"].min(), data["len"].max(), bins + 1)
                fig = px.histogram(
                    data,
                    x="len",
                    nbins=bins,
                    title="Распределение длин описаний",
                    template=theme
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges[0],
                        end=bin_edges[-1],
                        size=(bin_edges[1] - bin_edges[0]),
                    ),
                )
                fig.update_layout(
                    bargap=0.1,
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    title=dict(x=0.4)
                )
                fig.update_layout(
                    xaxis_title="Длина описания в символах",
                    yaxis_title="Частота",
                )
                st.plotly_chart(fig)

                st.subheader("Boxplot для длин описаний")
                fig = px.box(data, x="len", template=theme)
                fig.update_layout(
                    title=dict(text="Boxplot длин описаний", x=0.4),
                    xaxis=dict(showgrid=True, title="Длина описания в символах"),
                )
                st.plotly_chart(fig)

                st.subheader("Гистограмма количества эпитетов в описании*")
                st.write("*количество слов и словосочетаний, записанных через запятую, в описании изображения")

                bins6 = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_epitets_num"
                )
                bin_edges6 = np.linspace(
                    data["epitets_num"].min(), data["epitets_num"].max(), bins6 + 1
                )
                fig = px.histogram(
                    data,
                    x="epitets_num",
                    nbins=bins6,
                    title="Количество эпитетов в описании",
                    template=theme
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges6[0],
                        end=bin_edges6[-1],
                        size=(bin_edges6[1] - bin_edges6[0]),
                    ),
                )
                fig.update_layout(
                    bargap=0.1,
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    title=dict(x=0.4)
                )
                fig.update_layout(
                    xaxis_title="Количество эпитетов",
                    yaxis_title="Частота",
                )
                st.plotly_chart(fig)

                st.subheader("Boxplot для количества эпитетов")
                fig = px.box(data, x="epitets_num", template=theme)
                fig.update_layout(
                    title=dict(text="Количество эпитетов в описании", x=0.4),
                    xaxis=dict(showgrid=True, title="Количество эпитетов"),
                )
                st.plotly_chart(fig)

                st.title("Анализ изображений")
                value_counts = data["rgb"].value_counts()
                pie_data = pd.DataFrame({
                    "Value": ["RGB", "BW"],
                    "Count": [
                        value_counts.get("RGB", 0),
                        value_counts.get("BW", 0)
                    ]
                })
                st.subheader("Соотношение RGB и чёрно-белых логотипов")
                fig = px.pie(pie_data, names="Value", values="Count", template=theme)
                fig.update_layout(title=dict(text="Тип изображения", x=0.4))
                st.plotly_chart(fig)

                st.subheader("Высота изображений")
                bins1 = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_height"
                )
                bin_edges1 = np.linspace(data["h"].min(), data["h"].max(), bins1 + 1)
                fig = px.histogram(
                    data,
                    x="h",
                    nbins=bins1,
                    title="Распределение высоты изображений",
                    template=theme
                )
                fig.update_layout(
                    xaxis_title="Высота",
                    yaxis_title="Частота",
                    title=dict(x=0.4)
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges1[0],
                        end=bin_edges1[-1],
                        size=(bin_edges1[1] - bin_edges1[0]),
                    ),
                )
                st.plotly_chart(fig)

                st.subheader("Ширина изображений")
                bins2 = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_wight"
                )
                bin_edges2 = np.linspace(data["w"].min(), data["w"].max(), bins2 + 1)
                fig = px.histogram(
                    data,
                    x="w",
                    nbins=bins2,
                    title="Распределение ширины изображений",
                    template=theme
                )
                fig.update_layout(
                    xaxis_title="Ширина",
                    yaxis_title="Частота",
                    title=dict(x=0.4)
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges2[0],
                        end=bin_edges2[-1],
                        size=(bin_edges2[1] - bin_edges2[0]),
                    ),
                )
                st.plotly_chart(fig)

                st.subheader("Соотношение сторон")
                bins3 = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_ratio"
                )
                bin_edges3 = np.linspace(data["ratio"].min(), data["ratio"].max(), bins3 + 1)
                fig = px.histogram(
                    data,
                    x="ratio",
                    nbins=bins3,
                    title="Распределение соотношений сторон h/w",
                    template=theme
                )
                fig.update_layout(
                    xaxis_title="Соотношение сторон",
                    yaxis_title="Частота",
                    title=dict(x=0.4)
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges3[0],
                        end=bin_edges3[-1],
                        size=(bin_edges3[1] - bin_edges3[0]),
                    ),
                )
                st.plotly_chart(fig)

                st.subheader("Соотношение сторон (без квадратных)")
                filtered_data = data[data["ratio"] != 1]["ratio"]
                if not filtered_data.empty:
                    bins4 = st.slider(
                        "Количество интервалов (bins)", 5, 50, 10, key="bins_ratio1"
                    )
                    bin_edges4 = np.linspace(filtered_data.min(), filtered_data.max(), bins4 + 1)
                    fig = px.histogram(
                        filtered_data,
                        x="ratio",
                        nbins=bins4,
                        title="Распределение соотношений сторон h/w",
                        template=theme
                    )
                    fig.update_layout(
                        xaxis_title="Соотношение сторон",
                        yaxis_title="Частота",
                        title=dict(x=0.4)
                    )
                    fig.update_traces(
                        xbins=dict(
                            start=bin_edges4[0],
                            end=bin_edges4[-1],
                            size=(bin_edges4[1] - bin_edges4[0]),
                        ),
                    )
                    st.plotly_chart(fig)

                st.subheader("Количество пикселей")
                bins5 = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_pixel"
                )
                bin_edges5 = np.linspace(data["pixel"].min(), data["pixel"].max(), bins5 + 1)
                fig = px.histogram(
                    data,
                    x="pixel",
                    nbins=bins5,
                    title="Распределение количества пикселей",
                    template=theme
                )
                fig.update_layout(
                    xaxis_title="Количество пикселей",
                    yaxis_title="Частота",
                    title=dict(x=0.4)
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges5[0],
                        end=bin_edges5[-1],
                        size=(bin_edges5[1] - bin_edges5[0]),
                    ),
                )
                st.plotly_chart(fig)


# --- FastAPI ---
@fastapi_app.post("/generate")
async def generate_logo_api(label: str, apply_custom_lora: bool = False):
    try:
        base_image = generate_cgan_images(label, count=1)[0]
        upscaled = upscale_image(base_image, label)
        final_image = style_with_diffusion(upscaled, label, apply_custom_lora=apply_custom_lora)
        
        if not final_image or not os.path.exists(final_image):
            raise RuntimeError("Не удалось создать логотип")
            
        with open(final_image, "rb") as f:
            return {"status": "success", "image": base64.b64encode(f.read()).decode()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8001)

def main():
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    run_streamlit()

if __name__ == "__main__":
    # Отключаем oneDNN для TensorFlow чтобы избежать предупреждений
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()
