import requests
import json
from tqdm import tqdm

def get_vk_photos(user_id, vk_token):
    url = 'https://api.vk.com/method/photos.get'
    params = {
        'owner_id': user_id,
        'album_id': 'profile',
        'rev': 1,
        'count': 200,
        'access_token': vk_token,
        'v': '5.131'
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()['response']['items']

def upload_to_yandex_disk(file_path, yandex_token, file_name):
    url = f'https://cloud-api.yandex.net/v1/disk/resources/upload?path={file_path}&overwrite=true'
    headers = {
        'Authorization': f'OAuth {yandex_token}'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    upload_url = response.json()['href']
    
    with open(file_name, 'rb') as f:
        response = requests.put(upload_url, files={'file': f})
        response.raise_for_status()

def create_folder_on_yandex_disk(folder_name, yandex_token):
    url = f'https://cloud-api.yandex.net/v1/disk/resources'
    headers = {
        'Authorization': f'OAuth {yandex_token}'
    }
    params = {
        'path': folder_name
    }
    response = requests.put(url, headers=headers, params=params)
    response.raise_for_status()

def save_photos_info_to_json(photos_info, json_file):
    with open(json_file, 'w') as f:
        json.dump(photos_info, f, indent=4)

def main():
    user_id = input("Введите ID пользователя VK: ")
    vk_token = input("Введите токен VK: ")
    yandex_token = input("Введите токен Яндекс.Диска: ")
    
    # Получаем фотографии
    photos = get_vk_photos(user_id, vk_token)
    
    # Сортируем фотографии по размеру и лайкам
    sorted_photos = sorted(photos, key=lambda x: (-max(x['sizes'], key=lambda s: s['width'] * s['height'])['width'], x['likes']['count']))
    
    # Ограничиваем количество фотографий
    max_photos_count = 5
    selected_photos = sorted_photos[:max_photos_count]

    # Создаем папку на Яндекс.Диске
    folder_name = f'vk_photos_{user_id}'
    create_folder_on_yandex_disk(folder_name, yandex_token)

    photos_info = []
    
    for photo in tqdm(selected_photos):
        max_size_photo = max(photo['sizes'], key=lambda s: s['width'] * s['height'])
        likes_count = photo['likes']['count']
        date_uploaded = photo['date']
        
        file_name = f"{likes_count}_{date_uploaded}.jpg"
        file_path = f"{folder_name}/{file_name}"

        # Загружаем фотографию на Яндекс.Диск
        upload_to_yandex_disk(file_path, yandex_token, max_size_photo['url'])
        
        # Сохраняем информацию о загруженной фотографии
        photos_info.append({
            "file_name": file_name,
            "size": max_size_photo['type']
        })

    # Сохраняем информацию в JSON файл
    save_photos_info_to_json(photos_info, 'photos_info.json')

if __name__ == "__main__":
    main()