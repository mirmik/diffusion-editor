# Критическая стабилизация после архитектурного ревью

Дата: 2026-07-27  
Основание: [`architecture-review-2026-07-27.md`](architecture-review-2026-07-27.md)

## Итог

Критический пакет реализован в общем рабочем дереве. Переписывать редактор
не потребовалось: заложенные границы `EditorApplication` — `DocumentService`
— `LayerStack` — canvas/generation adapters выдержали усиление контрактов.

На момент финального прогона не осталось известных воспроизводимых
P0/P1-сценариев из scope карточек #899, #900, #902–#906, #908, #912,
#914, #915 и #917. Это не означает production-ready: dirty lifecycle,
autosave/recovery, delta history, общий resource scheduler и аппаратная
матрица всё ещё остаются отдельной работой.

## Что исправлено

### Native commands и canvas transactions — #899, #900

- Добавлен toolkit-neutral набор editor commands и production wiring
  Native root: history, clipboard, selection, layers и view используют один
  канонический handler path.
- Paint, erase, smudge, move, mask, mask erase и selection gesture создают
  ровно одну history entry.
- Незавершённый gesture привязан к стабильному layer ID; detach, capture loss,
  modal/document mutation и shutdown приводят к детерминированному rollback.
- В `DocumentService` добавлен общий mutation barrier. Команда, Undo/Redo,
  New/Open/Import сначала отменяют live edit, поэтому snapshot и gesture не
  могут вклиниться друг в друга.
- Mask eraser оформлен как compound mask+RGBA transaction: cancel и
  Undo/Redo восстанавливают и mask, и стёртый alpha.
- Ошибки tool begin/move/end, overlay/GPU callback и observer не оставляют
  активную сессию или захваченный pointer. Cleanup выполняется best-effort до
  конца и сохраняет исходную ошибку.
- Canvas gesture резервирует application document revision до первой прямой
  мутации, поэтому начатая ранее генерация становится stale сразу, а не
  только после записи history.

### Alpha contract и renderer — #902

- Public/document boundary теперь использует straight RGBA8, внутренний
  accumulator и cache — premultiplied float32.
- Исправлены повторное затемнение при flatten, nested solo и ancestor group
  opacity в prefix/exclude paths.
- Full и rect composite используют один CPU reference contract.
- GPU shader различает straight layer texture и premultiplied temporary group
  texture; устранён double-premultiply.
- Render cache ограничен memory/entry budget, учитывает aliases и использует
  LRU eviction.
- Закреплён линейный no-cache path для translucent stack: 15 слоёв дают 15
  чтений tile, а не экспоненциальный пересчёт.
- Native OpenGL smoke выполняет pixel parity check полупрозрачной группы.

### Commands, history и commit boundaries — #903

- `DocumentService` восстанавливает before-snapshot при падении action или
  post-action serialization.
- History переносит entry между Undo/Redo stacks только после успешного
  callback; неуспешный restore можно повторить.
- ZIP timestamp больше не создаёт ложную history entry для semantic no-op.
- Grid и generated-result paste валидируются до мутации; paste клипается по
  всем границам и собирается отдельно до commit.
- Ошибка presentation observer после успешно применённого snapshot/load уже
  не превращает commit в ложный failure и не пропускает остальные listeners.

### Immutable inference jobs и stale-result rejection — #904

- Все generation controllers используют immutable job context с `job_id`,
  document session/revision, layer ID, frozen mask/input/paste geometry,
  target pixel revision и tool/request fingerprint.
- Terminal result заново разрешает layer ID в текущем aggregate и
  отклоняется при смене session, revision, layer/tool, pixels, mask,
  geometry или request settings.
- New/Open/Import, delete, detach и shutdown инвалидируют связанные jobs.
- Result command владеет frozen pixels/provenance и не перечитывает mutable
  tool state при позднем применении.

### Grounding coordinates — #905

- Grounding DTO/result mapping использует canvas coordinates и canvas-sized
  selection.
- Smaller, offset и partially off-canvas layer больше не меняют геометрию
  результата и не приводят к shape exception.

### Atomic storage и archive hardening — #906

- Save выполняется во временный sibling, делает file fsync и atomic replace.
  Ошибка до replace сохраняет старый файл; temporary file удаляется.
- Ошибка directory fsync после уже состоявшегося replace логируется как
  durability warning, а не выдаётся за неуспешный Save.
- Load строит и валидирует detached aggregate и только затем одним commit
  заменяет текущий документ.
- Current-format manifest требует стабильные уникальные layer IDs, валидные
  active/solo references и поддерживаемые tool records. Legacy ID migration
  остаётся только для старых format versions.
- Проверяются canvas/layer/mask/selection/tool/source-patch shapes, dtypes,
  finite values, geometry и resource budgets. Save-side validation
  гарантирует, что созданный текущей версией архив можно загрузить обратно.
- NPY header разбирается до `np.load`: запрещены object/zero-size dtype,
  oversized/truncated payload и неожиданная mask/selection shape. Маленький
  ZIP с ложной гигантской shape больше не может вызвать предварительную
  многогигабайтную аллокацию.
- New/Open/Import сначала фиксируют core commit
  (session/history/project path), а ошибки settings/title/fit/status после
  него не оставляют смешанное состояние.

### Layer aggregate invariants — #908

- Root/children collections стали read-only views, attached structure меняет
  только `LayerStack`.
- Проверяются membership/ownership, unique IDs, parent consistency,
  duplicate reachability, cycles, foreign/stale targets, indices, last root,
  active и solo references.
- Save/load/serialization вызывают invariant validation; старый aggregate
  освобождает ownership после замены.

### Concurrency, Agent и lifecycle — #912, #915

- Worker queue сохраняет terminal ownership до публикации/poll terminal
  event; resubmit больше не может переставить завершения местами.
- Engine events и controllers коррелируют jobs по ID.
- Agent main-thread calls имеют generation-scoped state
  `pending/running/terminal`. Cancel/shutdown очищает ещё не запущенный
  callable, а retained callback старого dispatcher остаётся inert.
- Running callback является документированной атомарной границей:
  транзакция завершается, но cancelled request не получает success/tool
  output.

### Reproducibility и provenance — #914

- Seed `-1` разрешается один раз; Torch и NumPy используют только
  request-local generators. Одинаковые seed дают одинаковый preprocessing,
  глобальные RNG не изменяются.
- Добавлены versioned forward-compatible provenance DTO и canonical request
  fingerprint.
- Worker → engine → controller → history/tool → project roundtrip сохраняет
  resolved seed, request, input hashes, runtime versions, device/dtype и model
  identity.
- Для локальных artifacts вычисляется настоящий SHA-256; warn/strict policy
  различает immutable, floating, unknown и hash mismatch.
- Непинованные remote Instruct/IP-Adapter и LaMa честно остаются `floating`;
  strict policy их отклоняет, но код не выдумывает commit/hash.

### CPython 3.14t CI — #917

- Workflow больше не использует Python 3.10/py310 SDK.
- CI ожидает canonical py314t SDK asset + checksum, ставит зависимости через
  SDK-owned interpreter и запускает один `run-quality-gates.sh`.
- ABI, SOABI, disabled-GIL identity, pinned wheels, full tests и routine
  OpenGL/native smokes проверяются одним gate.
- Реальный GitHub Actions run остаётся внешней приёмкой: runner должен иметь
  опубликованный py314t SDK asset.

## Проверка

Финальная локальная матрица:

| Проверка | Результат |
|---|---|
| CPython | 3.14.6 free-threading, `cpython-314t`, GIL disabled |
| `git diff --check` | Пройден |
| `compileall` production/scripts/tests | Пройден |
| Последний полный unit/concurrency suite | `486 passed` |
| Unit/concurrency suite внутри OpenGL quality gate | `486 passed` |
| Main-process ABI/import/payload gate | Пройден |
| Legacy Termin OpenGL render smoke | Пройден, 3 frames |
| Production legacy startup smoke | Пройден, 3 frames |
| Native windowed OpenGL root smoke | Пройден, 3 frames, GPU group parity |

Команда успешного интеграционного прогона:

```sh
xvfb-run -a ./run-quality-gates.sh \
  --skip-resolver --skip-workers \
  --render-backend opengl --frames 3 --timeout 180
```

Попытка полного gate без `--skip-workers` дошла до worker stage после
зелёного suite, но локально остановилась из-за отсутствия
`.venv-workers/bin/python`. Это ограничение окружения, а не падение worker
теста. Network resolver также намеренно не повторялся.

## Что ещё не доказано и не исправлено

- Реальный Vulkan presentation/GPU compositor, Windows, CUDA/ROCm и настоящие
  большие ML-модели не проверялись.
- Remote model registry пока не выдаёт immutable revisions/hashes для всех
  backends; требуется отдельный verified asset registry.
- Нет dirty state, unsaved-changes prompt, autosave и crash recovery (#907).
- History всё ещё основана преимущественно на полных snapshots; нужны bounded
  deltas/coalescing (#910).
- Dense layer/mask storage и общий render/history memory budget остаются
  ограничением больших документов (#911).
- Canvas transaction пока делает полный before-copy image/mask в начале
  gesture; на больших слоях это даёт заметный peak memory (#900/#910/#911).
- Archive budgets осознанно остаются широкими (до 1 GiB на NPY и 8 GiB на
  проект), а legacy non-NPY payload читается eagerly. Для полностью
  недоверенных проектов нужна более строгая policy/streaming validation.
- Generated-result resize выполняется до проверки пересечения с target.
  Pathological runtime dimensions следует ограничить до resize.
- Лимит в 10 000 layers сочетается с несколькими рекурсивными traversal/
  serialization paths; экстремально глубокое дерево может получить
  `RecursionError`.
- `Layer.id`, `parent` и pixel array всё ещё публично изменяемы. Обычные UI и
  commands идут через aggregate, но произвольный внешний код способен обойти
  cache/invariant boundary (#908/#909).
- Нет общего inference/VRAM scheduler (#913).
- Typed revisioned `DocumentChange` stream (#909) всё ещё нужен: текущий
  monotonic application revision и mutation barrier закрывают correctness,
  но не заменяют полноценную change model.
- CI workflow требует подтверждения чистым GitHub Actions run; local
  `.venv-workers` и hardware gates должны быть подняты отдельно.

## Вывод по основе

Основа оказалась хорошей. Критические исправления в основном добавлялись в
существующие seams, а не обходили их:

- `DocumentService` стал транзакционной границей;
- `EditorApplication` — владельцем session/revision/jobs;
- `LayerStack` — владельцем дерева и persistence invariants;
- canvas adapters — источником intent, но не владельцем Undo semantics;
- worker boundary сохранил изоляцию ABI и тяжёлых ML dependencies.

Следующее разумное направление — не расширять число AI-панелей, а закончить
`DocumentChange`/dirty/recovery, delta history и model/resource registry.

## Состояние Kanboard

В карточки добавлены implementation/evidence/residual-risk комментарии.

- `On Test`: #899, #900, #902, #903, #904, #905, #906, #908, #912,
  #914 и #917.
- `Done`: #915.
- Umbrella #897 остаётся в Backlog до приёмки дочерних задач и выполнения
  следующих архитектурных этапов.

Карточки сознательно не закрыты автоматически: Vulkan/hardware, clean
GitHub Actions и ручная product acceptance остаются отдельными gates.
