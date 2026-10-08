"""Классификация тональности русскоязычных отзывов.
Запуск: python src/predict.py --model model
        python src/predict.py --model model --text "Ваш отзыв"
"""
import argparse
import torch
from transformers import pipeline

EXAMPLES = [
    "Потрясающий фильм! Сильный сюжет, отличная игра актёров, смотрел не отрываясь.",
    "Ужасное кино: скучный сюжет, пустые диалоги, два часа потраченного времени.",
    "Фильм неплохой, но ничего особенного. Один раз посмотреть можно.",
    "Ну, такое.",
    "Отличный фильм, если вы всегда мечтали выспаться в кинотеатре.",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model", help="папка с моделью или имя на HF Hub")
    parser.add_argument("--text", default=None, help="свой текст; без него берутся 5 примеров")
    args = parser.parse_args()

    clf = pipeline(
        "text-classification",
        model=args.model,
        tokenizer=args.model,
        top_k=None,                      # вероятности всех классов
        truncation=True,
        max_length=256,
        device=0 if torch.cuda.is_available() else -1,
    )

    texts = [args.text] if args.text else EXAMPLES
    for text, scores in zip(texts, clf(texts)):
        scores = sorted(scores, key=lambda s: s["score"], reverse=True)
        best = scores[0]
        print(f"\nТекст: {text}")
        print(f"Предсказание: {best['label']} ({best['score']:.1%})")
        print("Все классы: " + ", ".join(f"{s['label']}={s['score']:.2f}" for s in scores))

if __name__ == "__main__":
    main()