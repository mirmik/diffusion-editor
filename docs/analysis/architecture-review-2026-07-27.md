# Архитектурное ревью Diffusion Editor

Дата ревью: 2026-07-27  
Ревизия исходников на момент основной проверки: `240fe81` (`dev`)

> Статус после ревью: критический стабилизационный пакет реализован и
> проверен. Результат, границы исправлений и остаточные риски описаны в
> [`critical-fixes-2026-07-27.md`](critical-fixes-2026-07-27.md).

## Краткий вывод

Diffusion Editor не нуждается в переписывании с нуля. В проекте уже есть
хорошая исследовательская основа: доменная модель документа отделена от UI,
изменения оформляются командами, ML-нагрузка вынесена в отдельные процессы,
рендеринг декомпозирован, а тестовая база достаточно велика для текущего
размера проекта.

При этом редактор пока нельзя считать надёжным production-приложением.
Несколько важных контрактов существуют только частично:

- native UI не полностью подключён к history и application commands;
- публичный формат результата compositing не согласован по alpha;
- команды документа не атомарны при исключениях;
- асинхронная генерация привязана к mutable-объектам слоёв, а не к версии
  документа;
- сохранение, загрузка и закрытие документа пока не обеспечивают защиту
  пользовательских данных;
- история на полных снимках плохо масштабируется;
- критические инварианты дерева слоёв проверяются в UI, но не в доменном
  ядре.

Поэтому проект следует оценивать как сильный R&D-прототип и раннюю alpha,
а не как законченный редактор.

Ориентировочная оценка зрелости:

| Область | Оценка | Комментарий |
|---|---:|---|
| Архитектурное направление | 7/10 | Границы подсистем в основном выбраны правильно |
| Доменное ядро | 6/10 | Хорошая командная основа, но не хватает транзакций и инвариантов |
| ML/worker-архитектура | 7/10 | Удачная изоляция, слабая идентичность и координация jobs |
| Native-интеграция | 4/10 | Основные компоненты есть, production wiring не завершён |
| Надёжность пользовательских данных | 4/10 | Нет dirty lifecycle, recovery и атомарного I/O |
| Готовность продукта | 4/10 | Alpha; расширять feature surface пока рискованно |

## Методика ревью

Проверка состояла из четырёх частей.

### Статический анализ

Были просмотрены:

- composition root и жизненный цикл приложения;
- `LayerStack`, `Layer`, renderer, tile storage и history;
- команды и `DocumentService`;
- legacy и native UI;
- CPU- и GPU-compositor;
- diffusion, instruct, LaMa, segmentation и grounding controllers;
- ML workers и threaded lifecycle;
- persistence, настройки и agent tools;
- тесты, quality gates и CI.

Дополнительно была построена внутренняя карта импортов production-модулей.
В 108 Python-модулях и 256 внутренних связях циклы импорта не обнаружены.
Это сильный положительный сигнал: физическая структура пакетов пока не
превратилась в неразделимый монолит.

### Автоматические проверки

На локальном окружении выполнены:

| Проверка | Результат |
|---|---|
| `./venv/bin/python -m pytest -q` | `357 passed` |
| Основной CPython 3.14t ABI/import gate | Пройден |
| Native headless Vulkan smoke | Пройден с CPU compositor fallback |
| Native windowed OpenGL smoke | Пройден |
| Legacy OpenGL startup/render smoke | Пройден |
| Полный worker quality gate | Не выполнен: отсутствует `.venv-workers` |

### Целевые воспроизведения

Помимо тестов были выполнены небольшие воспроизведения для:

- потери native history после рисования;
- повторного затемнения полупрозрачного пикселя при `flatten`;
- частичной мутации при падении `DrawGridCommand`;
- Grounding на слое меньше canvas и с ненулевым offset;
- устаревшей ссылки на `Layer` после snapshot restore;
- создания циклического дерева через `move_layer`;
- стоимости snapshot history на документе 2048×2048;
- отсутствующих native command handlers.

### Ограничения

Ревью не доказывает корректность следующих путей:

- реальный windowed Vulkan GPU compositor: headless smoke переключился на
  CPU fallback;
- CUDA/ROCm и настоящие ML-модели;
- worker IPC в каноническом `.venv-workers`;
- Windows bootstrap и packaging;
- поведение на больших реальных проектах под длительной нагрузкой.

Проверка безопасности не была полноценным security-аудитом. Замечания по
секретам, remote agent и архивам следует считать архитектурным risk review.
Замер history является микробенчмарком и показывает порядок величины, а не
полный профиль производительности.

## Сильные стороны

### Разделение application, domain и presentation

[`EditorApplication`](../../diffusion_editor/app/application.py#L86) не
зависит напрямую от конкретного widget toolkit. Состояние представления
выведено в узкие presentation-порты, а ресурсы закрываются по явным фазам.
Это позволяет закончить native UI без переписывания модели документа и
generation controllers.

### Командная модель изменений

[`DocumentService`](../../diffusion_editor/document/document_service.py#L50)
даёт единый вход для типизированных команд и history. Текущая реализация
транзакций недостаточна, но сам seam выбран удачно: rollback, revisioning,
dirty state и change events можно добавить централизованно.

### Хорошая модульность

Document, canvas, rendering, generation, engines, workers и app разнесены по
пакетам. Request builders и result mappers отделены от UI. Это заметно
снижает стоимость тестирования и позволяет менять транспорт ML workers без
переноса бизнес-логики.

### Изоляция ML-процессов

Основной UI работает на free-threaded CPython 3.14t, а тяжёлые ML-зависимости
вынесены в отдельные процессы обычного CPython. Такая граница изолирует ABI,
падения native-библиотек и конфликтующие версии `torch`, `transformers` и
других пакетов.

Ценность process boundary выше, чем ценность самого no-GIL runtime. Даже
если позднее проект вернётся к обычному CPython для UI, worker-архитектуру
стоит сохранить.

### Рендеринг и инвалидация уже выделены в самостоятельную подсистему

Есть tile-oriented renderer, prefix caches, dirty rects, региональные GPU
uploads и отдельный compositor. Это хорошая база для дальнейшей оптимизации.
Важно лишь не путать tiled caching с sparse-хранением: сейчас изображение
слоя всё равно остаётся плотным массивом.

### Неплохая тестовая дисциплина

357 быстрых тестов, отдельные smoke-сценарии и документация quality gates —
существенное преимущество для молодого проекта. Документация не выдаёт
headless CPU fallback за настоящий Vulkan hardware gate.

## Подтверждённые дефекты и архитектурные риски

Приоритеты используются в следующем смысле:

- **P0** — блокирует native cutover или ломает базовую функцию редактора;
- **P1** — может повредить документ, дать неверный результат или создать
  серьёзную проблему производительности;
- **P2** — системный долг и риск дальнейшего развития;
- **P3** — локальный дефект с ограниченным влиянием.

## P0

### P0-1. Native Canvas изменяет документ без history

`NativeRoot` создаёт `NativeEditorCanvas` и coordinators, но не связывает
`EditorCanvasController.on_edit_begin/on_edit_end` с transaction/history
callbacks:
[`native_root.py:270–360`](../../diffusion_editor/app/native_root.py#L270).
Legacy UI выполняет это подключение явно:
[`editor_window.py:402–415`](../../diffusion_editor/app/editor_window.py#L402).

Воспроизведение:

1. открыть native root;
2. выполнить paint/erase/smudge/move либо редактирование mask;
3. убедиться, что пиксели изменились;
4. вызвать `undo()`;
5. `undo()` возвращает `None`, изменённые пиксели остаются.

Влияние: основная операция редактора визуально работает, но нарушает
ожидаемый контракт undo. Пользователь не может безопасно отменить жест.

Исправление:

- создать toolkit-neutral `CanvasEditCoordinator`;
- фиксировать слой, target, before state и dirty rect в момент начала жеста;
- завершать или отменять транзакцию при `pointer_up`, потере capture, смене
  документа и закрытии окна;
- использовать один coordinator в legacy и native UI;
- добавить integration-тест именно production composition root, а не
  изолированного controller с вручную подключёнными callbacks.

Критерий готовности: каждый mutating Canvas gesture создаёт ровно одну
history entry; undo/redo восстанавливают пиксели, mask, offset и selection.

### P0-2. Команды native shell показаны, но production handlers отсутствуют

[`NativeEditorView.activate_command()`](../../diffusion_editor/app/native_shell.py#L339)
возвращает `False`, когда handler не зарегистрирован. В production root
подтверждено отсутствие обработчиков для части базовых команд:

- undo/redo;
- copy/paste;
- selection commands;
- layer new/remove/flatten;
- view fit.

Изолированные тесты скрывают проблему, передавая handlers вручную.

Влияние: меню и shortcuts могут присутствовать в UI, но не выполнять
действие. Native UI нельзя считать функционально эквивалентным legacy.

Исправление:

- вынести app-owned `CommandCoordinator`;
- объявить единый registry `command_id → handler/state`;
- подключать один registry к обоим UI;
- добавить parity-тест, который проходит по всем enabled commands в
  production root и проверяет наличие handler;
- завершить чек-лист native parity до смены default UI.

## P1

### P1-1. Несогласованный alpha contract повреждает полупрозрачные пиксели

[`LayerRenderer._blend_image()`](../../diffusion_editor/document/layer_renderer.py#L356)
получает straight-alpha source, но сохраняет RGB результата в
premultiplied-виде. Публичный
[`LayerStack.composite()`](../../diffusion_editor/document/layer_stack.py#L409)
не обозначает этот формат, а
[`flatten()`](../../diffusion_editor/document/layer_stack.py#L297) передаёт
результат обратно в `Layer` как обычный RGBA.

Воспроизведение на одном пикселе:

```text
исходный straight RGBA       [255, 0, 0, 128]
composite                    [128, 0, 0, 128]
после первого flatten         [64, 0, 0, 128]
после второго flatten         [32, 0, 0, 128]
```

Влияние:

- повторный flatten затемняет изображение;
- экспорт полупрозрачного composite получает неверные RGB;
- возможны тёмные ореолы;
- CPU и GPU пути могут выдавать разные значения.

Исправление:

- зафиксировать канонический формат каждой границы;
- ввести хотя бы типизированные wrappers/документированные aliases
  `StraightRgba8` и `PremultipliedRgba8`;
- держать внутренний accumulator premultiplied, но unpremultiply ровно на
  display/export/flatten boundary;
- не использовать один `np.ndarray` как неявно оба формата.

Обязательные тесты:

- flatten idempotence;
- полупрозрачный PNG export;
- Porter–Duff reference cases;
- CPU/GPU parity;
- полностью прозрачный пиксель с ненулевым RGB;
- group opacity и несколько полупрозрачных слоёв.

### P1-2. `DocumentService` не откатывает частично выполненную команду

[`execute_snapshot_action()`](../../diffusion_editor/document/document_service.py#L75)
делает `before`, вызывает `action()` и только затем делает `after`. Если
`action()` падает, rollback отсутствует и history entry не создаётся.

Подтверждённый сценарий:

1. вызвать `DrawGridCommand(sections_x=2, sections_y=0)`;
2. vertical loop успевает записать пиксели;
3. horizontal loop делит на ноль в
   [`commands.py:198–230`](../../diffusion_editor/document/commands.py#L198);
4. документ частично изменён;
5. `history.can_undo == False`.

Agent schema также не задаёт `minimum: 1`:
[`agent/tools.py:459–475`](../../diffusion_editor/agent/tools.py#L459).

Похожий класс дефекта есть в generated-result paste: команда сначала
очищает слой, затем выполняет paste с координатами, которые не нормализованы:
[`commands.py:515–546`](../../diffusion_editor/document/commands.py#L515),
[`result_paste.py`](../../diffusion_editor/document/result_paste.py#L1).

Исправление:

- валидировать все инварианты до первой записи;
- выполнять команду внутри rollback-safe `DocumentTransaction`;
- при исключении восстанавливать `before`, не создавая history entry;
- различать validation error и internal command error;
- не полагаться на JSON schema как на единственную защиту.

Критерий готовности: для любой падающей команды состояние и revision до и
после вызова побитно/семантически эквивалентны.

### P1-3. Исключение в undo/redo уничтожает запись history

[`HistoryManager.undo()` и `redo()`](../../diffusion_editor/document/history.py#L85)
сначала удаляют entry из стека, затем вызывают callback. При исключении entry
не возвращается ни в исходный, ни в противоположный стек.

Исправление: перемещать entry только после успешного callback либо
гарантированно восстанавливать стек в `except`. Само применение snapshot
также должно быть транзакционным.

### P1-4. Асинхронная генерация адресует mutable `Layer`, а не документ

Generation controllers хранят прямые ссылки `_pending_layer` и
`_queued_layer`:
[`diffusion_controller.py:30–51`](../../diffusion_editor/generation/diffusion_controller.py#L30).
Тот же паттерн используется в instruct, LaMa, segmentation и grounding.

Snapshot restore вызывает
[`LayerStack.load_state()`](../../diffusion_editor/document/layer_stack.py#L547)
и создаёт новые Python-объекты слоёв. В результате прежний объект может иметь
тот же стабильный ID, но больше не принадлежать документу.

Подтверждённое воспроизведение:

1. запустить generation для слоя;
2. выполнить snapshot restore/undo;
3. получить новый объект текущего слоя с прежним ID;
4. дождаться результата;
5. controller возвращает старый отсоединённый объект.

Дополнительная проблема: `ApplyGeneratedResultCommand` читает `tool`,
`patch_*` и `mask` в момент завершения inference, а не использует frozen
контекст запроса. Изменение mask или patch rect во время inference меняет
смысл уже запущенной операции.

Рекомендуемый job contract:

```text
InferenceJob
  job_id
  document_session_id
  base_revision
  target_layer_id
  operation_kind
  immutable request
  frozen input / mask / paste_rect
  model identity / revision / hash
  seed
```

При завершении job должен заново разрешить `target_layer_id`, проверить
session/revision и применить явную conflict policy: `reject`, `rebase` либо
`create variant`. New/Open/Close/Remove Layer обязаны отменять или логически
инвалидировать связанные jobs.

### P1-5. Grounding падает на offset-слое меньше canvas

Grounding выполняется по composite всего canvas, но
[`map_grounding_result()`](../../diffusion_editor/generation/result_mapper.py#L73)
создаёт selection размером `layer.height × layer.width` и не переводит
canvas coordinates с учётом offset.

[`SetLayerSelectionCommand`](../../diffusion_editor/document/commands.py#L452)
справедливо требует mask размером всего canvas.

Воспроизведение:

1. canvas 8×8;
2. активный слой 4×4 с offset `(2, 2)`;
3. получить непустой grounding result;
4. mapper возвращает mask 4×4;
5. команда падает с
   `selection shape (4, 4) does not match canvas shape (8, 8)`.

Исключение проходит через poll application boundary и способно остановить
главный update loop.

Исправление:

- результат Grounding должен быть DTO в document/canvas coordinates;
- selection всегда должна иметь размер canvas;
- translation/clipping слоя выполнять явно;
- command failure на async boundary превращать в status/error event, а не
  выпускать в UI loop.

### P1-6. Нет безопасного жизненного цикла документа

В приложении не обнаружены:

- dirty/saved revision;
- запрос подтверждения при New/Open/Quit;
- autosave и crash recovery;
- восстановление незавершённой сессии.

[`save_project()`](../../diffusion_editor/document/layer_stack.py#L599)
открывает целевой ZIP сразу на запись. Ошибка процесса или питания может
уничтожить предыдущую исправную версию.

[`_load_from_zip()`](../../diffusion_editor/document/layer_stack.py#L487)
после частичного чтения очищает текущий stack и продолжает менять его.
Ошибка selection, manifest или cache rebuild способна оставить открытый
документ частично заменённым.

Исправление:

- ввести `DocumentSession` с `session_id`, URI, `revision`,
  `saved_revision`, dirty state и recovery metadata;
- сохранять во временный файл рядом с назначением;
- flush/fsync, затем `os.replace`;
- сначала загружать и полностью валидировать временный aggregate, затем
  одной операцией подменять текущий;
- ограничить размеры canvas, массивов, entries и общий uncompressed size
  архива;
- добавить checksums либо иную проверку целостности.

### P1-7. Доменное ядро допускает циклическое дерево слоёв

[`LayerStack.move_layer()`](../../diffusion_editor/document/layer_stack.py#L201)
не проверяет:

- `new_parent is layer`;
- перенос в собственного descendant;
- принадлежность обоих объектов stack;
- сохранение допустимого root tree.

Подтверждённый вызов переноса единственного root в самого себя создаёт
`layer.parent is layer`, добавляет слой в собственных children и оставляет
root list пустым. UI запрещает некоторые такие drag-and-drop операции, но
доменная команда и agent/API всё равно могут нарушить инвариант.

Исправление:

- сделать collections дерева закрытыми, наружу отдавать read-only views;
- валидировать membership и acyclic invariant в aggregate;
- запретить foreign objects;
- добавить property-based тесты случайных последовательностей add/remove/
  move/undo/redo;
- после каждой команды проверять дерево в debug/test builds.

### P1-8. Snapshot history не масштабируется с размером изображения

Почти каждая команда через `DocumentService` сериализует полный документ
до и после операции. Snapshot использует несжатый `ZIP_STORED`:
[`layer_stack.py:540–549`](../../diffusion_editor/document/layer_stack.py#L540).

Микробенчмарк: изменение opacity у одного слоя 2048×2048 сохранило около
32 MiB history. У одного 4096×4096 RGBA-слоя два снимка уже дают порядок
128 MiB на одну metadata-команду. Несколько слоёв и серия slider events
быстро расходуют гигабайты.

Лимит по умолчанию равен 5 GiB:
[`history.py:15–21`](../../diffusion_editor/document/history.py#L15), но
больший лимит не устраняет причину.

Исправление:

- inverse commands для metadata и структуры;
- command merge/coalescing для slider drags;
- tile/rect deltas для пикселей и mask;
- copy-on-write snapshots только для редких bulk operations;
- общий memory budget для history и renderer caches.

### P1-9. CPU partial compositor выдаёт неверный кадр

[`CanvasCompositeBridge._blend_layer_rect()`](../../diffusion_editor/canvas/canvas_composite.py#L191)
пересобирает регион из target layer и prefix ниже него, но:

- не добавляет слои выше target;
- не применяет opacity target layer;
- не воспроизводит полностью group semantics;
- трактует prefix buffer как straight RGB, хотя renderer возвращает
  premultiplied accumulator.

Влияние: после частичного stroke верхние слои могут временно исчезать в
регионе, opacity игнорируется, а CPU preview расходится с полным composite.

Исправление: partial compositing должен использовать один и тот же renderer
contract, что и full compositing. До реализации корректного пути безопаснее
делать full CPU rebuild для сложного дерева, чем показывать неверное
изображение.

### P1-10. Native pointer/edit session имеет ошибки жизненного цикла

Controller завершает жест только после `pointer_up`:
[`editor_canvas_controller.py:218–335`](../../diffusion_editor/canvas/editor_canvas_controller.py#L218).
Native widget не обеспечивает надёжный capture/cancel при release вне
Canvas. Воспроизведено состояние, когда edit session остаётся active.

Кроме того, `_move_tool_edit()` повторно читает
`layer_stack.active_layer`, а не использует слой, зафиксированный в начале
жеста. Смена active layer посреди stroke размазывает один жест по двум
слоям.

Исправление:

- захватывать pointer на press;
- иметь явные `commit_edit()` и `cancel_edit()`;
- завершать/cancel при capture loss, focus loss, detach и document swap;
- на протяжении жеста использовать `edit_session.layer`;
- запретить смену active target либо завершать текущий жест перед сменой.

### P1-11. CI противоречит заявленному runtime

Пакет требует Python 3.14:
[`pyproject.toml:9`](../../pyproject.toml#L9). При этом GitHub workflow
использует py310 SDK asset и Python 3.10:
[`ci.yml:12–33`](../../.github/workflows/ci.yml#L12).

Такой CI не проверяет фактическую архитектуру приложения и не может быть
надёжным release gate.

Исправление:

- запускать канонический SDK-owned CPython 3.14t;
- вызывать из CI тот же `run-quality-gates.sh`, что используется локально;
- добавить Windows bootstrap job;
- отдельно обозначить hardware/manual Vulkan gate;
- не считать headless CPU fallback GPU-проверкой.

## P2

### P2-1. Один mutable `LayerStack.on_changed` создаёт хрупкую цепочку

У `LayerStack` есть только один callback:
[`layer_stack.py:22`](../../diffusion_editor/document/layer_stack.py#L22).
Canvas, generation panels и layer tree последовательно заменяют его,
сохраняют предыдущий и пытаются восстановить при close.

Последствия:

- порядок подключения влияет на результат;
- один компонент может незаметно отключить остальные;
- исключение в одном callback блокирует downstream;
- teardown корректен только при обратном порядке;
- событие не сообщает revision, тип изменения и dirty rect.

Исправление: typed multi-subscriber event stream с unsubscribe token:

```text
DocumentChange {
    session_id,
    revision,
    kind: pixels | mask | structure | metadata | selection,
    layer_ids,
    dirty_rects
}
```

Этот механизм одновременно упростит dirty tracking, renderer invalidation,
history UI и native/legacy parity.

### P2-2. Два UI уже расходятся по семантике

Например, native solo проходит через undoable command, а legacy путь
изменяет `LayerStack` напрямую. В нескольких legacy panels остались прямые
мутации и доступ к приватному состоянию widgets.

Поддерживать две равноценные UI-реализации долго нецелесообразно.

Направление:

1. заморозить feature growth в legacy;
2. вынести все application actions в общие coordinators;
3. построить автоматизированную parity matrix;
4. принять native по явным критериям;
5. удалить legacy после короткого стабилизационного периода.

### P2-3. Composition roots становятся god objects

`EditorWindow`, `NativeRoot`, `GenerationPanelsCoordinator` и
`GPUCompositor` концентрируют слишком много wiring и policy. Это пока не
катастрофа — циклов импортов нет, — но добавление каждой новой модели
увеличивает fan-out и число ручных lifecycle-связей.

Нужны не новые абстракции «на всякий случай», а несколько конкретных
application services:

- `CommandCoordinator`;
- `CanvasEditCoordinator`;
- `DocumentSession`;
- `InferenceJobManager`;
- `RenderCoordinator`;
- `ModelAssetRegistry`.

### P2-4. Набор AI engines жёстко зашит

`EngineSet` и application poll/shutdown явно перечисляют каждый engine.
Добавление новой модели требует правок composition root, controller,
панелей и сериализации tool type.

Для редактора с заявленной расширенной поддержкой нейросетей нужен
типизированный capability registry:

```text
EngineDescriptor
  kind
  capabilities
  request_schema
  result_schema
  lifecycle
  resource_profile
```

Не следует сразу строить огромную plugin-платформу. Достаточно registry и
единого job/event contract, чтобы новые engines перестали расширять
`if/elif` во всём приложении.

### P2-5. Worker lifecycle и deadlines недостаточно строгие

[`ThreadedLifecycle`](../../diffusion_editor/engines/threaded_lifecycle.py#L60)
освобождает worker ownership перед публикацией terminal event. Контроллер
может запустить следующую операцию, и события разных поколений способны
наблюдаться в неожиданном порядке.

Также timeout отдельных этапов не всегда означает абсолютный deadline всей
операции.

Исправление:

- монотонный `job_id`/generation;
- упорядоченный terminal event;
- абсолютный deadline на job;
- игнорирование late events;
- единая политика cancel/shutdown;
- централизованный GPU/VRAM coordinator и backpressure.

### P2-6. Не вся генерация воспроизводима по seed

Основной torch generator seed-ится, но режим `latent_noise` использует
глобальный `np.random.randint`:
[`ml_backend.py:235–238`](../../diffusion_editor/workers/ml_backend.py#L235).
Одинаковый сохранённый seed поэтому не гарантирует одинаковый input.

Исправление: создавать локальный NumPy generator из того же operation seed
или сохранять отдельный derived seed. В provenance результата хранить model
revision/hash, pipeline options и все источники случайности.

### P2-7. Dense storage ограничивает большие документы

Каждый `Layer` создаёт полный
[`DenseTileGrid`](../../diffusion_editor/document/layer.py#L38).
`SparseTileGrid` существует, но production layers его не используют:
[`tiles.py:77–141`](../../diffusion_editor/document/tiles.py#L77).
Mask также хранится плотным массивом.

Tile renderer даёт хорошую инвалидацию, но пока не даёт sparse/out-of-core
память. После перехода history на deltas следует рассмотреть:

- sparse allocation прозрачных tiles;
- lazy mask tiles;
- cache LRU с единым memory budget;
- mmap/out-of-core backing для очень больших проектов;
- background compression только холодных tiles.

### P2-8. Agent имеет слишком широкие права и слабую модель секретов

Agent API key сохраняется через обычные settings:
[`dialogs.py:294–309`](../../diffusion_editor/app/dialogs.py#L294). Agent
получает mutating tools, включая удаление слоёв:
[`agent/tools.py:120–175`](../../diffusion_editor/agent/tools.py#L120), без
отдельного approval/preview.

Риски:

- секрет оказывается в JSON settings;
- удалённый endpoint может получить данные canvas;
- destructive tool call выполняется сразу;
- cancel не является гарантией, что уже поставленная mutation не будет
  обработана;
- сохранённые media-файлы не имеют ясной retention policy.

Исправление:

- OS keyring или environment secret;
- HTTPS для remote endpoint, исключение только для loopback;
- read-only tools по умолчанию;
- preview и подтверждение mutation batch;
- один undo transaction на подтверждённый batch;
- audit log с provider/request/job IDs;
- явное уведомление о передаче изображения и настройка очистки media.

### P2-9. Model assets недостаточно переносимы и воспроизводимы

Проект сохраняет абсолютные пути к моделям. Для переноса проекта на другой
компьютер и повторения результата этого недостаточно.

Целевой идентификатор:

```text
ModelAsset {
    provider,
    repository,
    revision,
    artifact_hash,
    local_override
}
```

Автоматические downloads должны быть pinned по revision и проверяться по
hash. В документе следует хранить provenance операции, а не только путь на
машине автора.

## P3

### P3-1. Settings могут сохраняться дважды

[`diffusion_editor/app/settings.py`](../../diffusion_editor/app/settings.py#L1)
вызывает `super().set()`, затем явный `save()`. Native Termin Settings уже
может сохранять значение внутри `set()`. Это лишний синхронный I/O и повод
уточнить единый контракт settings backend.

### P3-2. Метрики и quality-gate документация быстро устаревают

Некоторые документы всё ещё содержат прежнее число тестов. Автоматически
генерируемая короткая сводка runtime/ABI/test count уменьшит расхождение
между README, CI и локальными gates.

## Целевая архитектура

### Транзакционное ядро документа

Рекомендуемый основной поток:

```text
UI / Agent / Hotkey intent
            |
            v
    Application Command Coordinator
            |
            v
      immutable Command
            |
            v
      DocumentTransaction
      - validate
      - apply
      - rollback on error
      - increment revision
            |
            v
       DocumentChange
       /      |       \
  History  Renderer   Views
```

`LayerStack` остаётся aggregate, но:

- скрывает mutable collections;
- проверяет membership и tree invariants;
- не уведомляет UI одиночным callback;
- не знает о конкретных widgets;
- выпускает change set только после успешного commit.

### `DocumentSession`

Над `LayerStack` нужен объект с продуктовым состоянием:

```text
DocumentSession
  session_id
  uri
  document
  revision
  saved_revision
  dirty
  autosave_uri
  active_jobs
```

`New`, `Open`, `Close`, `Quit`, autosave и inference validity должны
опираться на session, а не на разрозненные поля UI.

### History

Рекомендуемая смешанная модель:

- inverse command для metadata и структуры;
- tile/rect delta для pixel/mask edits;
- coalescing непрерывного gesture/slider drag;
- copy-on-write snapshot только для редких bulk transforms;
- memory budget и telemetry.

### Pixel format

Нужно выбрать канонический public format. Практичный вариант:

- layer/source API: straight RGBA8;
- внутренний compositor accumulator: premultiplied float/UNORM;
- display/export/flatten/readback API: straight RGBA8;
- преобразование только на явно названной boundary.

Формат должен отражаться в именах функций, DTO или типизированных wrappers.

### Асинхронные AI-операции

Все engines должны работать через единый `InferenceJobManager`:

```text
submit immutable job
    -> queue / resource admission
    -> worker request(job_id)
    -> progress(job_id)
    -> terminal result(job_id)
    -> validate session/revision/layer
    -> apply, rebase, variant or reject
```

Это место должно владеть cancel, timeout, VRAM admission, late-result
filtering и provenance.

### Rendering

Renderer должен получать revisioned dirty change sets. Full и partial CPU
compositing, GPU compositor, export и flatten обязаны использовать единый
alpha/reference contract. У всех caches должен быть ограниченный budget и
наблюдаемая статистика hit/miss/memory.

### UI

Legacy и native UI должны быть тонкими adapters:

- отдают intent coordinators;
- отображают presentation state;
- не меняют `LayerStack` напрямую;
- не определяют undo semantics;
- не владеют inference identity.

После достижения parity legacy следует удалить, иначе каждая новая функция
будет требовать две реализации и два набора багов.

## Приоритетная дорожная карта

### Этап 0. Немедленная стабилизация

До добавления новых моделей и инструментов:

1. подключить native Canvas edits к history;
2. зарегистрировать все native production command handlers;
3. исправить alpha contract для composite/flatten/export;
4. добавить rollback при падении команды;
5. валидировать `DrawGrid` и generated-result paste;
6. исправить Grounding canvas coordinates;
7. запретить stale async result после undo/new/open/remove;
8. починить CI под CPython 3.14t.

Выходной критерий: ни один известный P0/P1 data-corruption сценарий не
воспроизводится, а для каждого добавлен regression test.

### Этап 1. Усиление ядра, 1–2 итерации

1. `DocumentSession`, dirty state и unsaved-changes flow;
2. атомарные save и transactional load;
3. typed `DocumentChange` event stream;
4. доменные инварианты дерева;
5. delta/inverse history и coalescing;
6. pointer capture и устойчивый Canvas edit lifecycle;
7. корректный partial compositor или безопасный full-rebuild fallback;
8. integration-тесты production composition roots.

### Этап 2. Завершение native cutover

1. единый `CommandCoordinator`;
2. автоматическая parity matrix legacy/native;
3. реальный windowed Vulkan audit на подходящем host;
4. профилирование idle rendering, dirty uploads и resize;
5. явная приёмка native;
6. перевод native в default;
7. удаление legacy после стабилизационного окна.

### Этап 3. Масштабируемая AI-архитектура

1. `InferenceJobManager`;
2. capability registry engines;
3. job deadlines, request IDs и late-event filtering;
4. model asset registry и provenance;
5. централизованный VRAM/GPU budget;
6. variants/rebase/bake/discard как недеструктивный AI workflow;
7. policy approvals и безопасное хранение agent secret.

### Этап 4. Большие документы и продуктовая эксплуатация

1. sparse/lazy tiles и masks;
2. общий cache/history memory budget;
3. autosave/recovery;
4. packaging и Windows gate;
5. telemetry производительности без пользовательских данных;
6. fault-injection тесты I/O, worker crash и process shutdown.

## Рекомендуемые архитектурные тесты

Помимо unit tests стоит добавить небольшой набор сквозных инвариантов:

- любой successful command увеличивает revision ровно один раз;
- failed command не меняет document hash и history;
- undo/redo exception не теряет entry;
- flatten дважды эквивалентен flatten один раз;
- CPU/GPU/full/partial composite совпадают в пределах оговорённого округления;
- дерево слоёв всегда ациклично, все узлы достижимы ровно один раз;
- Canvas gesture создаёт одну history entry независимо от количества move
  events;
- pointer capture loss либо commits, либо полностью rolls back gesture;
- late AI result не применяется к другой session/revision;
- Open повреждённого проекта не изменяет текущий документ;
- падение посередине Save сохраняет предыдущий файл;
- production native root имеет handlers для всех enabled commands.

## Как принимать дальнейшие архитектурные изменения

Новые функции желательно пропускать через четыре вопроса:

1. Какой immutable intent/command выражает изменение?
2. Как оно откатывается при исключении и через undo?
3. Какие revisioned change events оно выпускает?
4. Что происходит, если документ, слой или job изменились асинхронно?

Если на любой вопрос нет явного ответа, функцию лучше не подключать к UI до
уточнения контракта.

## Карточки Kanboard

<!-- KANBOARD_CARDS_BEGIN -->

Результаты ревью разложены на umbrella
[#897](http://localhost/task/897) и отдельные исполнимые карточки. Баги с
достаточно узким scope и воспроизведением помещены в `Ready`; крупные
архитектурные направления оставлены в `Backlog`.

| ID | Колонка | Размер | Карточка |
|---:|---|---|---|
| [#897](http://localhost/task/897) | Backlog | XL | `[architecture] Harden editor correctness and production readiness` |
| [#899](http://localhost/task/899) | Ready | M | `[native] Wire production editor command handlers` |
| [#900](http://localhost/task/900) | Ready | M | `[native/canvas] Make edit gestures transactional and capture-safe` |
| [#902](http://localhost/task/902) | Ready | L | `[render] Unify alpha representation and rect compositing` |
| [#903](http://localhost/task/903) | Ready | L | `[document] Make commands and history exception-safe` |
| [#904](http://localhost/task/904) | Ready | L | `[generation] Reject stale async results by session and revision` |
| [#905](http://localhost/task/905) | Ready | S | `[grounding] Build selections in canvas coordinates` |
| [#906](http://localhost/task/906) | Ready | M | `[storage] Make project save and load atomic` |
| [#907](http://localhost/task/907) | Backlog | L | `[document] Track dirty state and autosave recovery` |
| [#908](http://localhost/task/908) | Ready | M | `[layers] Enforce tree invariants in the domain model` |
| [#909](http://localhost/task/909) | Ready | M | `[document] Publish typed revisioned change events` |
| [#910](http://localhost/task/910) | Backlog | L | `[history/perf] Replace full snapshots with bounded deltas` |
| [#911](http://localhost/task/911) | Backlog | XL | `[storage/perf] Introduce COW tiles and bounded render caches` |
| [#912](http://localhost/task/912) | Ready | S | `[concurrency] Preserve terminal event order across resubmit` |
| [#913](http://localhost/task/913) | Backlog | L | `[ml] Add inference scheduling and VRAM budgets` |
| [#914](http://localhost/task/914) | Ready | M | `[generation] Make seeded runs reproducible and record provenance` |
| [#915](http://localhost/task/915) | Ready | S | `[agent] Drop queued mutations after cancellation` |
| [#916](http://localhost/task/916) | Backlog | L | `[agent/security] Protect secrets and approve mutations` |
| [#917](http://localhost/task/917) | Ready | S | `[ci] Align workflow with the CPython 3.14t quality gate` |

Typed blocker relations на доске:

- umbrella #897 заблокирован всеми перечисленными дочерними карточками;
- final native cutover #608 заблокирован #899, #900 и #902;
- runtime cutover gate #808 заблокирован regression-карточками #912 и #917;
- #907 зависит от #906 и #909;
- #910 зависит от #903 и #909;
- #911 зависит от #909 и #910;
- #913 зависит от #904.

В #608 и #808 добавлены комментарии с результатами аудита. В закрытой
карточке lifecycle #805 оставлена ссылка на regression #912.

<!-- KANBOARD_CARDS_END -->
