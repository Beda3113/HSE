import streamlit as st
import os
import zipfile
from datetime import datetime
from pathlib import Path

# Конфигурация страницы
st.set_page_config(
    page_title="Файловый хостинг",
    page_icon="📁",
    layout="wide"
)

# Настройки
UPLOAD_FOLDER = "public_files"
OUTPUT_FOLDER = "output"  # Папка, которую нужно скачивать
PUBLIC_URL = "http://87.228.39.18:8501"  # URL вашего сервера

# Создаем папки, если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def save_uploaded_file(uploadedfile):
    """Сохраняет загруженный файл и возвращает ссылку"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploadedfile.name}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(filepath, "wb") as f:
        f.write(uploadedfile.getbuffer())
    
    return f"{PUBLIC_URL}/{UPLOAD_FOLDER}/{filename}"

def create_zip_download(folder_path):
    """Создает zip-архив папки для скачивания"""
    zip_path = f"{folder_path}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    return zip_path

# Интерфейс
st.title("📁 Файловый хостинг и управление output")

tab1, tab2 = st.tabs(["Загрузка файлов", "Управление output"])

with tab1:
    with st.expander("ℹ️ Инструкция по загрузке"):
        st.write("""
        1. Загрузите файл с помощью формы ниже
        2. После обработки вы получите публичную ссылку
        3. Этой ссылкой можно делиться с другими пользователями
        """)

    uploaded_file = st.file_uploader("Выберите файл для публикации", type=None)

    if uploaded_file is not None:
        try:
            file_url = save_uploaded_file(uploaded_file)
            st.success("Файл успешно опубликован!")
            st.markdown(f"**Публичная ссылка:** [{file_url}]({file_url})")
            st.code(file_url, language="text")
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {e}")

    if os.path.exists(UPLOAD_FOLDER) and os.listdir(UPLOAD_FOLDER):
        with st.expander("📂 Ранее загруженные файлы"):
            files = os.listdir(UPLOAD_FOLDER)
            for file in files:
                st.markdown(f"- [{file}]({PUBLIC_URL}/{UPLOAD_FOLDER}/{file})")

with tab2:
    st.header("Управление папкой output")
    
    # Проверяем существование папки output
    if not os.path.exists(OUTPUT_FOLDER):
        st.warning(f"Папка {OUTPUT_FOLDER} не найдена!")
    else:
        # Показываем содержимое папки
        with st.expander(f"📂 Содержимое папки {OUTPUT_FOLDER}"):
            output_files = []
            for root, _, files in os.walk(OUTPUT_FOLDER):
                for file in files:
                    output_files.append(os.path.join(root, file))
            
            if not output_files:
                st.info("Папка output пуста")
            else:
                for file in output_files:
                    st.write(f"- {file}")
        
        # Кнопка для скачивания всей папки output
        if st.button("Скачать папку output как ZIP"):
            try:
                zip_path = create_zip_download(OUTPUT_FOLDER)
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Нажмите для скачивания",
                        data=f,
                        file_name="output.zip",
                        mime="application/zip"
                    )
                os.remove(zip_path)  # Удаляем временный zip-файл
            except Exception as e:
                st.error(f"Ошибка при создании архива: {e}")

# Запуск: streamlit run app.py
