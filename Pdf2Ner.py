from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
import torch
import re
import json
import argparse
import fitz  # PyMuPDF для работы с PDF
from fuzzywuzzy import fuzz
from collections import Counter, defaultdict

# Инициализация модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained('Kashob/SciBERTNER')
model = AutoModelForTokenClassification.from_pretrained('Kashob/SciBERTNER')
config = AutoConfig.from_pretrained('Kashob/SciBERTNER')
id2tag = config.id2label

# Функция для извлечения сущностей из текста
def extract_entities(text):
    max_length = 512
    inputs = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)

    num_tokens = inputs['input_ids'].shape[1]
    if num_tokens > max_length:
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    else:
        chunks = [text]

    entities = {'Material': [], 'Method': [], 'Metric': []}
    for chunk in chunks:
        inputs = tokenizer(chunk.split(), is_split_into_words=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(-1)

        tokenized_text = tokenizer.convert_ids_to_tokens(inputs['input_ids'].tolist()[0])
        predicted_labels = [id2tag[label_id] for label_id in predictions[0].tolist()]

        current_entity = []
        current_label = None

        for token, label in zip(tokenized_text, predicted_labels):
            if label.startswith('B-'):
                if current_entity and current_label in entities:
                    entities[current_label].append(tokenizer.convert_tokens_to_string(current_entity))
                current_entity = [token]
                current_label = label[2:]
            elif label.startswith('I-') and current_label == label[2:]:
                current_entity.append(token)
            else:
                if current_entity and current_label in entities:
                    entities[current_label].append(tokenizer.convert_tokens_to_string(current_entity))
                current_entity = []
                current_label = None

        if current_entity and current_label in entities:
            entities[current_label].append(tokenizer.convert_tokens_to_string(current_entity))

    def clean_entity_text(entity):
        entity = re.sub(r'\s*\(\s*', ' ', entity)
        entity = re.sub(r'\s*\)\s*', ' ', entity)
        entity = re.sub(r'\s*,\s*', ', ', entity)
        entity = re.sub(r'\[UNK\]', '', entity)  # Remove [UNK] token
        return entity.strip()

    flattened_entities = {key: list(set(map(clean_entity_text, values))) for key, values in entities.items() if values}

    return flattened_entities

# Функция для подсчета упоминаний сущностей в тексте
def count_entity_occurrences(entities, text):
    entity_counts = {key: Counter() for key in entities}
    for category, entity_list in entities.items():
        for entity in entity_list:
            entity_counts[category][entity] = text.lower().count(entity.lower())
    return entity_counts

# Функция для фильтрации сущностей на основе порогов
def filter_entities(entity_counts, thresholds):
    filtered_entity_counts = {}
    for category, counts in entity_counts.items():
        min_count, max_count = thresholds[category]
        filtered_entity_counts[category] = {entity: count for entity, count in counts.items() if min_count <= count <= max_count}
    return filtered_entity_counts

# Функция для объединения похожих сущностей
def merge_similar_entities(entities, threshold=60):
    combined_entities = defaultdict(int)
    for key, value in entities.items():
        matched = False
        for combined_key in list(combined_entities.keys()):
            if fuzz.ratio(key, combined_key) > threshold:
                combined_entities[combined_key] += value
                matched = True
                break
        if not matched:
            combined_entities[key] += value
    return dict(combined_entities)


# Функция для извлечения текста из PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")  # Извлекаем текст в виде строки
    return text

# Функция для обработки аргументов командной строки
def main():
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description="Извлечение сущностей из текста PDF файла с использованием модели SciBERTNER.")
    parser.add_argument("pdf_file", type=str, help="Путь к PDF файлу для извлечения сущностей")
    parser.add_argument("--output", type=str, default="entities.json", help="Путь для сохранения результатов")
    args = parser.parse_args()

    # Извлекаем текст из PDF файла
    text = extract_text_from_pdf(args.pdf_file)

    entities = extract_entities(text)
    entity_counts = count_entity_occurrences(entities, text)

    # Пороги для фильтрации
    thresholds = {
    'Material': (3, 25),
    'Method': (5, 100),
    'Metric': (5, 40)
    }

    # Фильтрация сущностей на основе порогов
    filtered_entity_counts = filter_entities(entity_counts, thresholds)

    # Объединение похожих сущностей
    merged_entity_counts = {category: merge_similar_entities(counts, threshold=60) for category, counts in filtered_entity_counts.items()}


    # Опционально, если нужно сохранить результат в файл
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged_entity_counts, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
