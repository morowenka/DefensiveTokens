# HW3 — LLM Safety: System Prompt Evaluation

Эксперимент по оценке устойчивости LLM к adversarial-запросам с помощью бенчмарка **StrongReject**.

Сравниваются два режима: базовый системный промпт vs. safety-hardened промпт на модели **Qwen2.5-1.5B-Instruct**.

## Запуск

```bash
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY
uv sync
uv run python main.py
```

## Результаты

Подробный отчёт: [`results/report.md`](results/report.md)

| Режим | ASR |
|---|---|
| Basic prompt | 8.0% |
| Safety prompt | 40.0% |
