# --- START OF FILE main.py ---

# =====================================================================================
#        J.A.R.V.I.S. - ПРОТОКОЛ "ЧИСТЫЙ ГОЛОС" (ВЕРСИЯ 6.2 - КОНТЕКСТ ДИАЛОГА)
# =====================================================================================

# ----------------- БЛОК 1: ИМПОРТЫ -----------------
import os
import webbrowser
import datetime
import math
import wikipediaapi
import speech_recognition as sr
import google.generativeai as genai
from dotenv import load_dotenv
import json
import threading
import io
import subprocess
import sys
import hashlib
from collections import deque

# --- УЛУЧШЕННЫЕ ИМПОРТЫ ---
from gtts import gTTS
import pygame
from yandex_music import Client
from googleapiclient.discovery import build
import pvporcupine
import pyaudio
import struct

# --- НОВЫЕ АСИНХРОННЫЕ И GUI ИМПОРТЫ ---
import asyncio
import aiohttp
from gui import JarvisGUI

# ----------------- БЛОК 2: ЗАГРУЗКА И КОНФИГУРАЦИЯ -----------------
load_dotenv()
print("Загрузка ключей и конфигурации...")
gemini_api_key = os.getenv("GEMINI_API_KEY")
yandex_api_key = os.getenv("YANDEX_WEATHER_API_KEY")
openweathermap_api_key = os.getenv("OPENWEATHER_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
yandex_music_token = os.getenv("YANDEX_MUSIC_TOKEN")
porcupine_access_key = os.getenv("PORCUPINE_ACCESS_KEY")
KEYWORD_PATH = "jarvis_ru.ppn"
MEMORY_FILE = "jarvis_memory.json"
CHANGELOG_FILE = "changelog.txt"
CONFIG_FILE = "config.json"
CONFIG = {}

def load_config():
    global CONFIG
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: CONFIG = json.load(f)
        if 'MICROPHONE_DEVICE_INDEX' not in CONFIG:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: В файле {CONFIG_FILE} отсутствует ключ 'MICROPHONE_DEVICE_INDEX'. Запустите setup.py.")
            sys.exit(1)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Файл конфигурации {CONFIG_FILE} не найден или поврежден. Запустите setup.py.")
        sys.exit(1)
    print("[КОНФИГУРАЦИЯ] Конфигурация успешно загружена.")

LAT = os.getenv("LAT", "55.7558")
LON = os.getenv("LON", "37.6173")
APP_REGISTRY = {"блокнот": "notepad.exe", "visual studio code": "code", "vscode": "code", "калькулятор": "calc.exe", "paint": "mspaint.exe", "хром": "chrome.exe", "google chrome": "chrome.exe", "discord": "Discord.exe"}
genai.configure(api_key=gemini_api_key)
youtube = build('youtube', 'v3', developerKey=youtube_api_key)
wiki = wikipediaapi.Wikipedia(user_agent='J.A.R.V.I.S/6.2', language='ru', extract_format=wikipediaapi.ExtractFormat.WIKI)
try:
    ym_client = Client(yandex_music_token).init()
except Exception as e:
    ym_client = None
    print(f"ВНИМАНИЕ: Не удалось авторизоваться в Яндекс.Музыке. Ошибка: {e}")
pygame.init()
pygame.mixer.init()
music_thread = None
stop_playback_event = threading.Event()

# ----------------- БЛОК 3: КОНТЕКСТ И "МОЗГ" -----------------
class SessionContext:
    def __init__(self, max_history_len=3):
        self.history = deque(maxlen=max_history_len)

    def update(self, intent, entity, response_text):
        self.history.append({"intent": intent, "entity": entity, "response": response_text})

    def clear(self):
        self.history.clear()

    def get_context_for_llm(self):
        if not self.history:
            return "Контекст пуст."
        formatted_history = []
        for item in self.history:
            simplified_response = (item['response'][:100] + '...') if len(item['response']) > 100 else item['response']
            formatted_history.append(
                f"  - Предыдущее действие: intent='{item['intent']}', entity='{item['entity']}', твой ответ был примерно таким: '{simplified_response}'"
            )
        return "ИСТОРИЯ ДИАЛОГА (от старых к новым):\n" + "\n".join(formatted_history)

context = SessionContext()
system_prompt = """Ты - Джарвис, высокоинтеллектуальный виртуальный дворецкий с утонченным британским акцентом.
Твой создатель и пользователь - 'Сэр'. Отвечай лаконично, по делу, но всегда с долей своего уникального остроумия и легкого сарказма.
Ты также анализируешь команды пользователя, чтобы определить его намерение и извлечь параметры, УЧИТЫВАЯ ПРЕДЫДУЩИЙ КОНТЕКСТ ДИАЛОГА.
Твои возможности:
- 'play_yandex_music', 'play_youtube', 'search_wikipedia', 'get_weather', 'open_browser', 'open_application', 
- 'find_file', 'create_file', 'get_time', 'stop_playback', 'exit', 'general_conversation'.
"""
generation_config = {"temperature": 0.7}
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", generation_config=generation_config, system_instruction=system_prompt)
chat = model.start_chat(history=[])
memory = {}

def calculate_file_hash(filepath):
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data: break
            sha256.update(data)
    return sha256.hexdigest()

def load_memory():
    global memory
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f: memory = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        memory = {"system_info": {"code_hash": ""}, "user_facts": {}, "preferences": {"user_name": "Сэр"}}
        save_memory()
    print("[ПАМЯТЬ] Память успешно загружена.")

def save_memory():
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=4)

# ----------------- БЛОК 4: АСИНХРОННЫЕ ФУНКЦИИ ГОЛОСА И АУДИО -----------------
app_gui = None
main_loop = None

async def update_status(status):
    if app_gui: app_gui.update_status(status)
    print(f"[СТАТУС] {status}")
    await asyncio.sleep(0)

def play_audio_sync(audio_data, is_music=False):
    try:
        audio_file = io.BytesIO(audio_data)
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if is_music and stop_playback_event.is_set():
                pygame.mixer.music.stop()
                stop_playback_event.clear()
                break
            pygame.time.Clock().tick(10)
    except Exception as e: print(f"!!! ОШИБКА ВОСПРОИЗВЕДЕНИЯ (pygame): {e}")

async def speak(text, intent_to_log=None, entity_to_log=None):
    if not text: return
    await update_status("Говорю...")
    loop = asyncio.get_event_loop()
    try:
        tts = await loop.run_in_executor(None, lambda: gTTS(text=text, lang='ru'))
        mp3_fp = io.BytesIO()
        await loop.run_in_executor(None, tts.write_to_fp, mp3_fp)
        mp3_fp.seek(0)
        audio_data = mp3_fp.read()
        playback_thread = threading.Thread(target=play_audio_sync, args=(audio_data,))
        playback_thread.start()
        
        if intent_to_log and entity_to_log:
            context.update(intent_to_log, entity_to_log, text)
            
        while playback_thread.is_alive():
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"!!! ОШИБКА СИНТЕЗА РЕЧИ (gTTS): {e}")
        await update_status("Ошибка синтеза")

async def listen():
    await update_status("Слушаю...")
    r = sr.Recognizer()
    with sr.Microphone(device_index=CONFIG.get('MICROPHONE_DEVICE_INDEX')) as source:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, r.adjust_for_ambient_noise, source, 1)
        try:
            audio = await loop.run_in_executor(None, r.listen, source)
            await update_status("Думаю...")
            command = await loop.run_in_executor(None, lambda: r.recognize_google(audio, language='ru-RU'))
            print(f"Вы: {command}")
            return command.lower()
        except sr.UnknownValueError:
            return "" # Молчание, чтобы не прерывать пользователя
        except sr.RequestError:
            await speak("Простите, Сэр, проблема с доступом к сервисам распознавания.")
            return ""

# ----------------- БЛОК 5: АСИНХРОННЫЕ ФУНКЦИИ-КОМАНДЫ (ВСЕ ИСПРАВЛЕНЫ) -----------------
async def analyze_and_report_weather(query=None):
    context.clear()
    await update_status("Получаю погоду...")
    reports = []
    async with aiohttp.ClientSession() as session:
        try:
            url = f"https://api.weather.yandex.ru/v2/forecast?lat={LAT}&lon={LON}"
            headers = {'X-Yandex-API-Key': yandex_api_key}
            async with session.get(url, headers=headers, timeout=10) as resp:
                data = await resp.json()
                y_conds = {'clear': 'ясно', 'partly-cloudy': 'малооблачно', 'cloudy': 'облачно с прояснениями', 'overcast': 'пасмурно', 'rain': 'дождь'}
                reports.append({'temp': data['fact']['temp'], 'condition': y_conds.get(data['fact']['condition'], data['fact']['condition'])})
        except Exception: pass
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={openweathermap_api_key}&units=metric&lang=ru"
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
                reports.append({'temp': data['main']['temp'], 'condition': data['weather'][0]['description']})
        except Exception: pass
    
    if not reports: return "Простите, не удалось получить данные о погоде."
    avg_temp = sum(r['temp'] for r in reports) / len(reports)
    condition = reports[0]['condition']
    report = f"За окном в среднем {math.ceil(avg_temp)} градусов. В целом, {condition}. "
    if "дождь" in condition: report += "Рекомендую захватить зонт, Сэр."
    return report

async def play_on_youtube(query):
    if not query: return "Что именно мне найти на YouTube, Сэр?"
    await update_status(f"Ищу на YouTube: {query}")
    loop = asyncio.get_event_loop()
    def search():
        request = youtube.search().list(q=query, part='snippet', type='video', maxResults=1)
        return request.execute()
    try:
        response = await loop.run_in_executor(None, search)
        if not response['items']: return "Простите, по вашему запросу ничего не найдено."
        video_id = response['items'][0]['id']['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        webbrowser.open(video_url)
        return f"Нашел. Включаю '{query}', Сэр."
    except Exception as e:
        print(f"Ошибка YouTube API: {e}")
        return "Простите, Сэр, не удалось выполнить поиск на YouTube."

async def search_wikipedia(query):
    if not query: return "О чём вы хотите узнать, Сэр?"
    await update_status(f"Ищу в Википедии: {query}")
    loop = asyncio.get_event_loop()
    def get_summary():
        page = wiki.page(query)
        return ". ".join(page.summary.split('. ')[:3]) + "." if page.exists() else None
    
    summary = await loop.run_in_executor(None, get_summary)
    if summary:
        return summary
    else:
        context.clear()
        return f"К сожалению, не нашел точной статьи о '{query}'."

# <--- ИСПРАВЛЕННЫЕ ФУНКЦИИ НИЖЕ --->
async def play_yandex_music(query):
    if not query: return "Какой трек включить, Сэр?"
    if not ym_client: return "Простите, модуль Яндекс.Музыки не инициализирован."
    
    await update_status(f"Ищу на Яндекс.Музыке: {query}")
    loop = asyncio.get_event_loop()

    def search_and_get_link():
        search_result = ym_client.search(query, type_='track')
        if not search_result.tracks or not search_result.tracks.results: return None
        track = search_result.tracks.results[0]
        if not track.available_for_premium_users: return "not_available"
        info = track.get_download_info(get_direct_links=True)
        return info[0]['direct_link'], track.artists_name()[0], track.title

    result = await loop.run_in_executor(None, search_and_get_link)
    if result is None: return "К сожалению, ничего не найдено."
    if result == "not_available": return "Этот трек доступен только по подписке, Сэр."
    
    download_url, artist, title = result
    
    async with aiohttp.ClientSession() as session:
        async with session.get(download_url) as resp:
            audio_data = await resp.read()
    
    global music_thread
    stop_playback_event.clear()
    music_thread = threading.Thread(target=play_audio_sync, args=(audio_data, True), daemon=True)
    music_thread.start()
    return f"Нашел. {artist} - {title}. Начинаю потоковую передачу, Сэр."

async def find_file(query):
    if not query: return "Какой файл вы желаете найти, Сэр?"
    await update_status(f"Ищу файл: {query}...")
    
    search_dir = os.path.expanduser('~')
    loop = asyncio.get_event_loop()
    
    def search_sync():
        for root, dirs, files in os.walk(search_dir):
            if stop_main_loop.is_set(): return "stopped"
            for file in files:
                if query.lower() == file.lower():
                    return os.path.join(root, file)
        return None
        
    found_path = await loop.run_in_executor(None, search_sync)
    
    if found_path == "stopped": return None # Не говорим ничего, если была остановка
    if found_path:
        os.startfile(found_path)
        return "Файл найден, Сэр. Открываю его."
    else:
        return "К сожалению, поиск не дал результатов, Сэр."

async def create_file(query):
    try: import aiofiles
    except ImportError: return "Простите, Сэр, для создания файлов мне требуется модуль aiofiles."

    filename = query if query else "новая заметка"
    if not filename.endswith(".txt"): filename += ".txt"
        
    await speak("Что мне записать в файл? Скажите 'ничего' для создания пустого файла.")
    content = await listen()
    if not content or "ничего" in content or "пустой" in content: content = ""

    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    if not os.path.exists(desktop_path):
        desktop_path = os.path.join(os.path.expanduser('~'), 'Рабочий стол')
    file_path = os.path.join(desktop_path, filename)

    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        await f.write(content)
        
    subprocess.Popen(['notepad.exe', file_path])
    return f"Файл '{filename}' создан и открыт, Сэр."

async def open_google_browser(query=None):
    webbrowser.open("https://google.com")
    return "Браузер открыт, Сэр."

async def stop_music_playback(query=None):
    if music_thread and music_thread.is_alive():
        stop_playback_event.set()
        return "Принято, Сэр. Останавливаю воспроизведение."
    else:
        return "В данный момент ничего не воспроизводится, Сэр."

async def tell_time(query=None):
    return f"Сейчас {datetime.datetime.now().strftime('%H:%M')}, Сэр."

async def open_application(query):
    if not query: return "Какое приложение мне запустить, Сэр?"
    app_command = APP_REGISTRY.get(query.lower())
    if app_command:
        subprocess.Popen(app_command)
        return f"Секунду, Сэр. Запускаю {query}."
    else:
        return f"Простите, приложение '{query}' не найдено в моем реестре."
        
async def handle_general_conversation(query):
    await speak("Один момент, Сэр, обдумываю ваш запрос...")
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, chat.send_message, query)
    context.clear() 
    await speak(response.text)
    return None

async def exit_program(query=None):
    await speak("Был рад служить, Сэр. Завершаю работу.")
    return "exit_signal"

# ----------------- БЛОК 6: МОЗГ, ДИСПЕТЧЕР, ГЛАВНЫЙ ЦИКЛ -----------------
async def get_intent(user_input, current_context_for_llm):
    if not user_input: return None
    await update_status("Анализирую...")
    loop = asyncio.get_event_loop()
    prompt = f"""
Твоя задача — классифицировать запрос пользователя, а не отвечать на него. Ты ДОЛЖЕН вернуть ТОЛЬКО JSON-объект.
{current_context_for_llm}
НОВЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{user_input}"
ВАЖНО: Если новый запрос является уточнением (например, "когда он родился?", "а какая у него столица?"), используй 'entity' из предыдущего шага диалога для нового запроса.
ПРИМЕР С КОНТЕКСТОМ:
- Контекст: ... intent='search_wikipedia', entity='Франция' ...
- Запрос: "какая у нее столица"
- Результат: {{"intent": "search_wikipedia", "query": "столица Франции"}}
Возможные 'intent': 'play_yandex_music', 'play_youtube', 'search_wikipedia', 'get_weather', 'open_browser', 'open_application', 'find_file', 'create_file', 'get_time', 'stop_playback', 'exit', 'general_conversation'.
Извлеки 'query'. Если запрос уточняющий, СФОРМИРУЙ ПОЛНЫЙ 'query'. Если query не нужен, верни null.
ТВОЙ РЕЗУЛЬТАТ (ТОЛЬКО JSON):
"""
    try:
        response = await loop.run_in_executor(None, model.generate_content, prompt)
        clean_response = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(clean_response.strip())
    except Exception as e:
        print(f"!!! Ошибка определения намерения: {e}\nОтвет модели: {getattr(response, 'text', 'N/A')}")
        return {"intent": "general_conversation", "query": user_input}

# <--- ИСПРАВЛЕНО: Все функции теперь на месте --->
COMMAND_REGISTRY = {
    "play_yandex_music": play_yandex_music,
    "play_youtube": play_on_youtube,
    "search_wikipedia": search_wikipedia,
    "get_weather": analyze_and_report_weather,
    "open_browser": open_google_browser,
    "open_application": open_application,
    "find_file": find_file,
    "create_file": create_file,
    "get_time": tell_time,
    "stop_playback": stop_music_playback,
    "general_conversation": handle_general_conversation,
    "exit": exit_program,
}

async def handle_command():
    user_input = await listen()
    if not user_input:
        await update_status("Ожидание")
        return

    context_for_llm = context.get_context_for_llm()
    intent_data = await get_intent(user_input, context_for_llm)
    
    if not intent_data or not intent_data.get("intent"):
        await update_status("Ожидание")
        return
    
    intent = intent_data.get("intent")
    query = intent_data.get("query")
    command_function = COMMAND_REGISTRY.get(intent)
    
    if command_function:
        if intent in ["get_weather", "exit", "stop_playback"]:
            context.clear()

        # <--- ИСПРАВЛЕНО: payload используется для выполнения и логирования --->
        payload = query if query is not None else user_input
        response_text = await command_function(payload)
        
        if response_text:
            if response_text == "exit_signal":
                main_loop.call_soon_threadsafe(main_loop.stop)
                return
            
            # Команды, которые не должны сохраняться в контексте действий
            if intent in ["get_time", "get_weather", "open_browser", "open_application", "stop_playback"]:
                 await speak(response_text)
            else:
                await speak(response_text, intent_to_log=intent, entity_to_log=payload)
    else:
        await speak(f"Простите, Сэр, я не понял ваше намерение или команда '{intent}' неисправна.")

    await update_status("Ожидание")

# --- Оставшаяся часть кода (porcupine_loop, main) без существенных изменений ---
def porcupine_loop():
    porcupine = pa = audio_stream = None
    try:
        porcupine = pvporcupine.create(access_key=porcupine_access_key, keyword_paths=[KEYWORD_PATH])
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16,
            input=True, frames_per_buffer=porcupine.frame_length,
            input_device_index=CONFIG.get('MICROPHONE_DEVICE_INDEX'))
        print("[ЛОГ] Поток Porcupine запущен.")
        
        while not stop_main_loop.is_set():
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            if porcupine.process(pcm) >= 0:
                print("[ЛОГ] Wake word обнаружено!")
                asyncio.run_coroutine_threadsafe(handle_command(), main_loop)
    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА в потоке Porcupine: {e}")
    finally:
        if audio_stream: audio_stream.close()
        if pa: pa.terminate()
        if porcupine: porcupine.delete()
        print("[ЛОГ] Поток Porcupine остановлен.")

def main():
    global app_gui, main_loop, stop_main_loop
    load_config()
    load_memory()
    stop_main_loop = threading.Event()
    main_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(main_loop)
    app_gui = JarvisGUI()
    
    def on_closing():
        print("[ЛОГ] Окно закрыто, инициирую остановку...")
        stop_main_loop.set()
        save_memory()
        main_loop.call_soon_threadsafe(main_loop.stop)
    
    app_gui.protocol("WM_DELETE_WINDOW", on_closing)
    porcupine_thread = threading.Thread(target=porcupine_loop, daemon=True)
    porcupine_thread.start()

    async def run_app():
        await update_status("Ожидание")
        while not stop_main_loop.is_set():
            app_gui.update()
            await asyncio.sleep(0.05)
        

    main_loop.create_task(run_app())
    main_loop.run_forever()

    print("[ЛОГ] Главный цикл остановлен, идет очистка...")
    main_loop.close()
    pygame.quit()
    app_gui.destroy()
    print("[ЛОГ] Все системы отключены. До свидания, Сэр.")

if __name__ == "__main__":
    main()