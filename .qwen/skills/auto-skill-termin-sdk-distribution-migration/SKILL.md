---
name: termin-sdk-distribution-migration
description: Как актуализировать diffusion-editor под переименованные/перестроенные Python-дистрибутивы Termin SDK (например, tcbase→termin-base, tgfx→termin-graphics-core, tmesh→termin-mesh) и починить ошибку "python-runtime-manifest.json does not describe required distribution".
source: auto-skill
extracted_at: '2026-09-06T22:33:19.893Z'
---

# Миграция под переименованные дистрибутивы Termin SDK

Когда `./install-deps.sh` падает с
`ERROR: python-runtime-manifest.json does not describe required distribution '<name>'`,
либо проект нужно синхронизировать с новой версией Termin SDK, которая
переименовала/перестроила Python-пакеты.

## Как диагностировать

1. Найди путь к SDK: `$TERMIN_SDK` → `.termin-sdk` (файл в корне проекта) → `/opt/termin`.
2. Прочитай `<sdk>/python-runtime-manifest.json` (schema 4) — там актуальные имена
   дистрибутивов и `python_abi` (version/soabi/free_threaded/py_gil_disabled).
3. Посмотри wheelhouse `<sdk>/wheels/*.whl`. Для каждого нового wheel выложи
   содержимое (`python3 -c "import zipfile,glob; ..."`), чтобы увидеть реальный
   layout модулей. Имя wheel-файла использует **underscore** (`termin_base-*.whl`),
   хотя дистрибутив называется через дефис (`termin-base`).
4. Составь mapping «старое имя → новое имя → путь модуля». Пример для SDK 0.5.2:
   - `tcbase` → `termin-base` (модуль `termin.base`, в т.ч. `termin.base.settings`,
     `termin.geombase._geom_native`)
   - `tgfx` → `termin-graphics-core` (модуль `termin.graphics`,
     `termin.graphics._graphics_native`)
   - `tmesh` → `termin-mesh` (модуль `termin.mesh`)
5. Проверь, что `termin` — **PEP 420 namespace package** (в wheel нет `termin/__init__.py`),
   поэтому `import termin.base` работает без явного `__init__`.
6. Убедись, что **C-символы не менялись** (например `tc_shader_ensure_tgfx2`,
   `Tgfx2Context`) — меняются только Python-пути модулей, а не нативный API.

## Что менять

1. `diffusion_editor/sdk_runtime.py`:
   - `DIRECT_TERMIN_DISTRIBUTIONS` — новые имена.
   - Проверка нативного build ID для schema 3 — новые имена.
   - `verify_imports` — `import termin.base`, `import termin.graphics`,
     `from termin.graphics import Tgfx2Context, configure_default_shader_runtime`.
2. Все Python-импорты (`from tcbase import X` → `from termin.base import X`,
   `from tgfx import X` → `from termin.graphics import X`,
   `from tmesh import X` → `from termin.mesh import X`). Затрагивает engines,
   workers, generation-контроллеры, canvas, app, multiview_studio, grounding,
   automation.
3. `requirements-project.txt` — новые имена дистрибутивов.
4. Тесты: `tests/test_sdk_runtime.py`, `tests/test_dependency_contract.py`,
   и все тесты, где были `from tmesh import ...`.
5. `install-deps.sh` — расширь cleanup-строку `pip uninstall --yes ...` на все
   **старые** имена (`tcbase tgfx tmesh tcgui`), чтобы при переустановке не
   остались устаревшие пакеты в venv. `tcgui` — отдельный пакет, только в cleanup.
6. Текстовые сообщения об ошибках, где фигурировало старое имя (например
   "published no tmesh for UUID" → "published no mesh").

## Подводные камни (проверено на практике)

- `termin_requirement_closure()` возвращает имена **в алфавитном порядке**
  (`sorted`). Если в тесте жёсткий tuple-сравнение — порядок должен совпадать с
  сортировкой, а не с порядком в `DIRECT_TERMIN_DISTRIBUTIONS`. После переименования
  порядок меняется (например `termin-graphics-core` теперь идёт **перед**
  `termin-gui-native`).
- Глоб по wheel-файлу в тестах — через underscore: `termin_base-*.whl`,
  **не** `termin-base-*.whl`.
- Используй `./venv/bin/pytest` (Python 3.14, есть `tomllib`). Системный
  `python3` (3.10) не имеет `tomllib` и не поднимет `test_dependency_contract.py`.
- После правок: `./install-deps.sh` (пересобирает venv с новым SDK-билдом и
  чистит старые пакеты), затем полный `./venv/bin/pytest`.
- Оставшиеся в venv пакеты от **другого** SDK-билда, не используемые проектом
  (например `termin-image`), не трогай — они не в closure.
- Не переименовывай `tcgui` — он не входит в closure, только в uninstall-cleanup.

## Финальная проверка

- `./install-deps.sh` проходит до конца (исходная ошибка исчезла).
- Полный `./venv/bin/pytest` зелёный.
- Смоук-импорты: `import termin.base, termin.graphics, termin.mesh` + ключевые
  модули приложения.
- `pip list` в venv: старые имена (`tcbase`/`tgfx`/`tmesh`) отсутствуют, новые
  `termin-*` стоят на актуальном SDK-билде.
