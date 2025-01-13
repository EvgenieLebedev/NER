# Решение задачи распознавания именованных сущностей для научных статей

Постановка задачи: дан текст научной статьи **на английском языке** , необходимо сформировать список, состощий из 4 видов сущностей
1) Решаемые задачи
2) Методы и алгоритмы
3) Наборы данных 
4) Метрики оценки

В репозитории представлено два модуля, позволяющие сформировать файл `entities.json`. Приведем пример данного json

```json

{
    "Material": "salinas dataset, indian pines",
    "Method": "3 - d convolutional neural networks, prclstm, prclstm model",
    "Metric": "aa, oa, kappa coefficient, average accuracy, te, overall accuracy, test error",
    "Task": "3 - d structural processing, spectral - spatial classification of hyperspectral imageries hsi"
}

```

Представлено две реализации модуля: 
- Без необходимости извлечения текста (Text2Ner)
- Обрабатывающий pdf файлы для извлечения текста (Pdf2Ner)

```cmd

python text.py "Hyperspectral images provide a precise representation of the earth's surface, with abundant spectral and spatial features,  but normal classification algorithms use only the information provided"

```

```cmd

python  Pdf2Ner.py "De Backer и др. - 2005 - A Band Selection Technique for Spectral Classification.pdf"

```
P.S: Не рекомендуется давать слишком большие тексты, поскольку количество мусорных ответов прямопорпоционально объему. Рекомендуемый режим использования -> обработка глав Results или Conclusion или раздела Abstact 
