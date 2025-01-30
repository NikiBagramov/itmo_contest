import os
import re
import requests
import feedparser
import threading
import queue
from flask import Flask, request, jsonify
from duckduckgo_search import ddg

app = Flask(__name__)

# Очередь запросов
request_queue = queue.Queue()

# API-ключи (замените на свои)
OPENAI_API_KEY = "your-openai-api-key"
DEEPSEEK_API_KEY = "your-deepseek-api-key"
LLAMA_API_URL = "your-llama-api-url"

# Количество потоков-воркеров
NUM_WORKERS = 10  # Можно увеличить до 100 при необходимости


# -------------------------------------------------
# 1. Поиск информации в DuckDuckGo
# -------------------------------------------------
def search_links(query: str, max_links: int = 3):
    results = ddg(query + " site:itmo.ru", region='wt-wt', safesearch='moderate', max_results=max_links)
    return [{"title": r["title"], "link": r["href"], "snippet": r.get("body", "")} for r in results] if results else []


# -------------------------------------------------
# 2. Получение новостей из RSS-ленты ИТМО
# -------------------------------------------------
def fetch_latest_news(max_news: int = 3):
    rss_url = "https://news.itmo.ru/ru/rss/"
    feed = feedparser.parse(rss_url)
    return [{"title": entry.title, "link": entry.link, "summary": entry.summary} for entry in feed.entries[:max_news]]


# -------------------------------------------------
# 3. Функции работы с LLM (GPT, DeepSeek, LLaMA)
# -------------------------------------------------
def ask_chatgpt(prompt: str):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4",
        "messages": [{"role": "system", "content": "You are an AI assistant."},
                     {"role": "user", "content": prompt}],
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No answer found.")


def ask_deepseek(prompt: str):
    url = "https://api.deepseek.com/v1/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "prompt": prompt,
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json().get("choices", [{}])[0].get("text", "No answer found.")


def ask_llama(prompt: str):
    url = LLAMA_API_URL
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json().get("text", "No answer found.")


# -------------------------------------------------
# 4. Объединение контекста из поиска и новостей
# -------------------------------------------------
def combine_context(user_query, search_results, news_results):
    context = []

    if search_results:
        context.append("🔍 **Поиск в Интернете:**")
        for item in search_results:
            context.append(f"- [{item['title']}]({item['link']}): {item['snippet']}")

    if news_results:
        context.append("\n📰 **Новости:**")
        for item in news_results:
            context.append(f"- [{item['title']}]({item['link']}): {item['summary']}")

    return "\n".join(context)


# -------------------------------------------------
# 5. Определение правильного варианта ответа
# -------------------------------------------------
def parse_multiple_choice(query: str):
    pattern = r"(\d+)\.\s*(.*)"
    lines = query.split("\n")
    choices = []
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            num = int(match.group(1))
            text = match.group(2)
            choices.append((num, text))
    return choices


def match_answer_to_choice(answer_texts, choices):
    if not choices:
        return None

    answer_votes = {num: 0 for num, _ in choices}

    for answer_text in answer_texts:
        ans_lower = answer_text.lower()
        for num, choice_text in choices:
            overlap = sum(1 for token in choice_text.lower().split() if token in ans_lower)
            if overlap > 0:
                answer_votes[num] += overlap

    best_choice = max(answer_votes, key=answer_votes.get) if max(answer_votes.values()) > 0 else None
    return best_choice


# -------------------------------------------------
# 6. Основная обработка запроса
# -------------------------------------------------
def process_request(request_data):
    request_id = request_data["id"]
    user_query = request_data["query"]

    choices = parse_multiple_choice(user_query)
    search_results = search_links(user_query, max_links=3) if "итмо" in user_query.lower() else []
    news_results = fetch_latest_news() if "новост" in user_query.lower() else []
    context_text = combine_context(user_query, search_results, news_results)

    gpt_response = ask_chatgpt(f"{user_query}\n\nКонтекст:\n{context_text}")
    deepseek_response = ask_deepseek(f"{user_query}\n\nКонтекст:\n{context_text}")
    llama_response = ask_llama(f"{user_query}\n\nКонтекст:\n{context_text}")

    final_answer = max(set([gpt_response, deepseek_response, llama_response]), key=[gpt_response, deepseek_response, llama_response].count)

    best_choice_num = match_answer_to_choice([final_answer], choices)

    response_data = {
        "id": request_id,
        "answer": best_choice_num,
        "reasoning": f"Сводный ответ: {final_answer}",
        "sources": [item["link"] for item in search_results + news_results][:3]
    }

    return response_data


# -------------------------------------------------
# 7. Рабочий поток (обрабатывает запросы в порядке очереди)
# -------------------------------------------------
def worker():
    while True:
        request_data = request_queue.get()
        if request_data is None:
            break  # Остановка потока

        response = process_request(request_data)
        print(f"Обработан запрос {response['id']}: {response}")  # Логирование
        request_queue.task_done()


# -------------------------------------------------
# 8. Основной эндпоинт API /api/request
# -------------------------------------------------
@app.route("/api/request", methods=["POST"])
def handle_request():
    data = request.get_json(force=True)
    request_queue.put(data)  # Добавляем запрос в очередь
    return jsonify({"status": "queued", "id": data["id"]})


# -------------------------------------------------
# 9. Запуск рабочих потоков
# -------------------------------------------------
threads = []
for _ in range(NUM_WORKERS):
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    threads.append(thread)


# -------------------------------------------------
# 10. Запуск Flask-сервера
# -------------------------------------------------
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
    finally:
        for _ in threads:
            request_queue.put(None)  # Завершаем потоки
        for thread in threads:
            thread.join()
