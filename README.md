# Решение задачи распознавания именованных сущностей для научных статей

Постановка задачи: дан текст научной статьи **на английском языке** , необходимо сформировать список, состощий из 4 видов сущностей
1) Решаемые задачи
2) Методы и алгоритмы
3) Наборы данных 
4) Метрики оценки

В репозитории представлено два модуля, позволяющие сформировать файл `entities.json`. Приведем пример данного json

```json

{
    "Material": {
        "hsi data": 16,
        "training and testing samples": 3,
        "university": 8,
        "paviau dataset": 3,
        "hyperspectral data": 12,
        "indian pines": 26,
        "lett": 9
    },
    "Method": {
        "recurrent analysis": 49,
        "svm": 17,
        "cnn2": 115,
        "pca": 55,
        "prclstm": 112,
        "convolutional neural network": 51,
        "neural net": 37,
        "framework": 11,
        "dimensionality": 7,
        "rnn": 21,
        "dropout": 7,
        "methods": 23,
        "emp": 7,
        "deep learning": 8,
        "deep": 21,
        "clstm layer": 7,
        "bn": 16,
        "rf": 38
    },
    "Metric": {
        "stm validation loss": 34,
        "average": 6,
        "loss": 22,
        "κ": 11,
        "computational": 25,
        "oa": 23,
        "dimensionality": 7,
        "accuracy": 11,
        "dropout": 7,
        "aa": 10,
        "statistical": 8
    }
}

```

Представлено две реализации модуля: 
- Без необходимости извлечения текста (Text2Ner)
- Обрабатывающий pdf файлы для извлечения текста (Pdf2Ner)

```cmd

python text.py "Hyperspectral images provide a precise representation of the earth's surface, with abundant spectral and spatial features,  but normal classification algorithms use only the information provided"

```

```cmd
python Pdf2Ner.py path/to/your/pdf_file.pdf --output path/to/save/entities.json
```

P.S: requiments.txt  составлены под то, что используется gpu, с cuda-драйверами верссии 11.8
