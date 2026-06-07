# pipeline/ — Оптимизация системного промпта через gepa

Пайплайн автоматической оптимизации системного промпта для чат-ботов (разблокировка карт, ВиС, КИБ). Использует библиотеку [`gepa`](https://github.com/gepa-ai/gepa) (`gepa.optimize()`) для рефлексивной мутации промптов с Pareto-отбором кандидатов.

## Архитектура

```
pipeline/
├── run.py               # Точка входа — gepa.optimize() + тестовая оценка
├── eval_best.py         # Оценка лучших промптов из артефактов на test-датасетах
├── adapter.py           # UnblockCardAdapter — разблокировка карт (single-component)
├── kib_adapter.py       # KIBAdapter — КИБ, классификация (single-component)
├── vis_adapter.py       # VisAdapter — ВиС, маршрутизация документов (multi-component)
├── llm.py               # LLM-адаптеры (OpenAI SDK, langchain-gigachat), конфиг провайдеров
├── config.py            # Pydantic-модели + загрузка YAML-конфигов
├── schema.py            # Валидация датасетов
├── report.py            # Генерация heatmap Excel, тестовые отчёты
├── collect_results.py   # Сбор run_summary.json в experiments.xlsx
├── configs/
│   ├── unblock/         # Конфиги для разблокировки карт
│   ├── kib/             # Конфиги для КИБ
│   └── vis/             # Конфиги для ВиС
├── prompts/
│   ├── seed.md          # Начальный системный промпт (unblock)
│   ├── vis_seed.md      # Seed-промпт для ВиС
│   └── reflection_prompt_template.yaml  # Шаблон рефлексии-мутатора
├── ВиС/                 # Скрипты/датасеты ВиС (датасеты не в git — PII)
│   └── prepare_datasets.py  # Подготовка train/val сплитов для ВиС
└── requirements.txt
```

## Провайдеры LLM

Два типа провайдеров в `llm.py`:

- **`openai_compatible`** — через OpenAI Python SDK (`openai.OpenAI`). Для Qwen3-32B, gpt-oss-120b и др. моделей на vLLM.
- **`gigachat`** — через `langchain-gigachat`. Для GigaChat-2-Max/Pro/Light с mTLS-аутентификацией.

Параметры модели задаются в `extra_params` конфига и передаются как `extra_body` (OpenAI) или kwargs (GigaChat).

## Требования

- Python 3.11+
- `pip install -r requirements.txt` (gepa, openai, langchain-gigachat, openpyxl, pydantic, pyyaml)
- Доступ к GigaChat API (mTLS сертификаты в `.certs/`)
- Датасеты в `datasets/` (не включены в репозиторий)

## Быстрый старт

```bash
# Unblock
python run.py --config configs/unblock/task-max_reflect-max.yaml

# КИБ
python run.py --config configs/kib/kib_max_max.yaml

# ВиС (сначала подготовить датасеты)
python ВиС/prepare_datasets.py
python run.py --config configs/vis/vis_max_max.yaml

# Переоценить лучшие промпты из артефактов на test-датасетах
python eval_best.py --configs-dir configs/unblock

# Собрать результаты
python collect_results.py --artifacts-dir artifacts/gepa_pipeline
```

## CLI-параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--config` | *обязательный* | Путь к YAML-конфигу эксперимента |
| `--max-calls` | из конфига | Override `optimization.max_calls` |
| `--patience` | из конфига | Override `optimization.patience` |

## YAML-конфигурация

Пример конфига с разными task/reflection провайдерами:

```yaml
run_name: task-GigaChat-2-Max_reflect-qwen32b

_gigachat: &gigachat
  type: gigachat
  base_url: https://gigachat-ift.sberdevices.delta.sbrf.ru/v1
  timeout_seconds: 120
  verify_ssl: false
  cert_file: .certs/cert_agents.txt
  key_file: .certs/key_agents.txt
  extra_params:
    profanity_check: false
    repetition_penalty: 1

_openai: &openai
  type: openai_compatible
  timeout_seconds: 120
  verify_ssl: true

task_provider:
  <<: *gigachat
  name: gigachat-max
  model: GigaChat-2-Max

reflection_provider:
  <<: *openai
  name: qwen32b
  model: qwen32b
  base_url: http://10.27.60.9:2717/v1
  api_key_env: INTERNAL_LLM_API_KEY
  extra_params:
    chat_template_kwargs:
      enable_thinking: true
    top_k: 20
    max_tokens: 32768

mutator:
  type: gepa_default
  temperature: 0.7
  reflection_prompt_template: |
    ...

optimization:
  max_calls: 3000
  patience: 30
  minibatch_size: 15
  candidate_selection_strategy: pareto
  use_merge: false
  eval_temperature: 0.0
  seed: 0

train:
  - datasets/train.json
val:
  - datasets/val.json
test: datasets/test_orig.json
seed_prompt: prompts/seed.md
heatmap_template: error_heatmap.xlsx
run_dir: artifacts/gepa_pipeline
```

## Выходные артефакты

| Файл | Описание |
|---|---|
| `best_prompt.txt` | Лучший найденный системный промпт |
| `events.jsonl` | Лог итераций оптимизации (gepa) |
| `candidate_tree.html` | Визуализация дерева кандидатов (gepa) |
| `experiments.xlsx` | Сводная таблица экспериментов |
| `heatmap_*.xlsx` | Хитмапы ошибок по тестовым датасетам |
| `token_usage.json` | Счётчики токенов (сохраняются между перезапусками) |
| `run_summary.json` | Полная сводка прогона |

Метрика — accuracy = доля успешных шагов / всего шагов (0.0–1.0): regex-проверки для unblock, корректность классификации для КИБ, точность маршрутизации для ВиС.
