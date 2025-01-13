from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
import torch
import re
import json
import argparse

# Инициализация модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained('Kashob/SciBERTNER')
model = AutoModelForTokenClassification.from_pretrained('Kashob/SciBERTNER')
config = AutoConfig.from_pretrained('Kashob/SciBERTNER')
id2tag = config.id2label

# Функция для извлечения сущностей из текста
def extract_entities(text):
    # Токенизация текста
    inputs = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(-1)

    # Преобразование токенов в текст и предсказания меток
    tokenized_text = tokenizer.convert_ids_to_tokens(inputs['input_ids'].tolist()[0])
    predicted_labels = [id2tag[label_id] for label_id in predictions[0].tolist()]

    # Сбор сущностей по категориям
    entities = {'Material': [], 'Method': [], 'Metric': [], 'Task': []}
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

    # Очистка текста сущностей
    def clean_entity_text(entity):
        entity = re.sub(r'\s*\(\s*', ' ', entity)
        entity = re.sub(r'\s*\)\s*', ' ', entity)
        entity = re.sub(r'\s*,\s*', ', ', entity)
        return entity.strip()

    # Убираем дубликаты и создаем финальный вывод
    flattened_entities = {key: ', '.join(set(map(clean_entity_text, values))) for key, values in entities.items() if values}
    
    return flattened_entities

# Функция для обработки аргументов командной строки
def main():
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description="Извлечение сущностей из текста с использованием модели SciBERTNER.")
    parser.add_argument("text", type=str, help="Текст для извлечения сущностей")
    args = parser.parse_args()

    # Извлекаем сущности
    result = extract_entities(args.text)

    # Печать результатов
    print(json.dumps(result, ensure_ascii=False, indent=4))

    # Опционально, если нужно сохранить результат в файл
    with open('entities.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
