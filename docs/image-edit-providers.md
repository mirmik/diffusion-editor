# Image Edit providers

AI Edit — один доменный инструмент и один generation lifecycle для моделей,
которые принимают исходное изображение и текстовую инструкцию. Модель не
кодируется в типе слоя, controller или native UI.

## Контракт

`ImageEditProfile` описывает устойчивый ID, provider, model identity и полный
список параметров. Каждый `ImageEditParameter` содержит тип, диапазон,
значение по умолчанию и признак load-time параметра. Native-панель строит
объединение зарегистрированных схем: параметры не скрываются в `Advanced` или
сворачиваемых секциях; параметры текущего профиля включены, остальные видны и
отключены. Это сохраняет стабильную раскладку и явно показывает различия
моделей.

`InstructTool` сохраняет исторический `tool_type = "instruct"` для чтения
старых проектов, но хранит `model_profile_id` и отдельный словарь значений для
каждого профиля. Новые AI Edit layers используют Qwen по умолчанию. Старые
проекты без новых полей мигрируют в профиль `instruct-pix2pix`.

`ImageEditRequest` передаёт worker только изображение, stable profile ID и
effective parameters. Worker выполняет `load_image_edit` и `image_edit`, а
seed, model identity, profile ID и полный effective parameter set попадают в
provenance. Смена prompt/steps не перезагружает модель; смена model, revision,
dtype, device, offload, VAE tiling или состав стека LoRA вызывает reload.

LoRA не являются фиксированными scalar-параметрами профиля. Каждый AI Edit
tool хранит отдельный упорядоченный стек для каждого profile ID. Строка стека
содержит устойчивый ID, отображаемое имя, source (локальный путь либо Hugging
Face repository), enabled и weight. Native-панель позволяет добавлять и
удалять строки, менять порядок, путь, включение и вес; фиксированного числа
слотов нет. Стек сохраняется в `.deproj`, входит в provenance и load identity.

Каждая строка также показывает каталог `Installed LoRA`. Он рекурсивно
сканирует `~/soft/ComfyUI/models/loras`, Forge `models/Lora`, соседние
`Lora`/`loras` относительно настроенного каталога checkpoints и дополнительные
пути из setting `lora_dirs` или `DIFFUSION_EDITOR_LORA_DIRS` (разделитель —
системный `PATH`). Символические ссылки на один файл дедуплицируются, служебная
`.cache` пропускается. `Custom / Hugging Face…` сохраняет ручной source для
repo ID и нестандартных путей.

## Встроенные профили

- `qwen-image-edit-2511` — `QwenImageEditPlusPipeline`, основной профиль.
  По умолчанию используются 4 шага при найденной Lightning LoRA, иначе 40.
- `qwen-image-edit-2511-multiple-angles` — тот же Qwen pipeline с отдельным
  адаптером
  [`fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA).
  Начальный стек содержит Lightning и Multiple Angles с независимыми весами
  `1.0` и `0.9`; пользователь может изменить или дополнить его.
- `flux2-klein-4b` — `Flux2KleinPipeline`, быстрый distilled профиль с четырьмя
  шагами.
- `sensenova-u1.5-8b-mot-preview` — standalone `sensenova_u1` image-to-image
  runtime с отдельными config/tokenizer и GGUF checkpoint. Локальный Q8 профиль
  по умолчанию использует проверенные 8 шагов и output budget 1 MP.
- `instruct-pix2pix` — совместимый legacy-профиль; он не удалён.

Production worker импортирует только Diffusers/Transformers и не импортирует
ComfyUI. Проверенные ComfyUI workflows остаются reference fixtures для
сверки параметров и визуального parity.

## Формат локальных весов

Параметр `Model / local directory` принимает Hugging Face model ID либо готовую
Diffusers pipeline directory. Upstream pipeline directories велики: Qwen и
FLUX включают transformer, text encoder, tokenizer/processor, VAE и scheduler.

Qwen дополнительно принимает отдельные `Transformer checkpoint` и
`Text encoder checkpoint`. Если установленные ComfyUI scaled-FP8 файлы найдены,
они подставляются по умолчанию и загружаются standalone-адаптером без импорта
ComfyUI. FP8 Linear-веса остаются сжатыми при хранении и деквантизуются по одному
слою для BF16-вычисления. Precision-sensitive BF16/F32 параметры сохраняют
исходный dtype. Upstream `Model` при этом остаётся источником конфигурации,
processor, tokenizer, scheduler и VAE. Чтобы вернуться к полному upstream BF16,
нужно очистить оба component checkpoint поля.

Multiple Angles автоматически ищется в
`~/soft/ComfyUI/models/loras/qwen-image-edit-2511-multiple-angles-lora.safetensors`.
Другой default задаётся через
`DIFFUSION_EDITOR_QWEN_MULTIPLE_ANGLES_LORA`. Prompt начинается с `<sks>` и
затем задаёт азимут, высоту камеры и дистанцию именно в таком порядке, например
`<sks> front-left quarter view elevated shot medium shot`. Обычный Qwen-профиль
этот адаптер не загружает.

Старые документы с плоскими `lora_path`, `lora_scale`, `angle_lora_path` и
`angle_lora_scale` мигрируют в две строки стека при чтении. После следующего
сохранения используются только `profile_lora_adapters`; старые поля обратно не
записываются.

Для локального Qwen по умолчанию включён `Component CPU offload`: text encoder
и transformer последовательно занимают GPU, а не находятся там одновременно.
Это соответствует автоматической перестановке компонентов в проверенном
ComfyUI workflow, но не включает послойную выгрузку transformer.

SenseNova загружается через официальный `sensenova_u1`, а не через ComfyUI
custom node. Параметр `Config / tokenizer directory` указывает на локальную
директорию U1.5 либо Hugging Face ID, а `GGUF checkpoint` — на фактические
квантованные веса. Доступны `full`, `fast`, `balanced` и `low` VRAM modes из
официального layer-offload runtime. GGUF identity и effective parameters
попадают в provenance.

Upstream metadata `sensenova-u1` фиксирует reference Torch/Pillow строже, чем
общий worker runtime. Поэтому пакет устанавливается без зависимостей в
provider overlay внутри `.venv-workers`; совместимые `gguf`, `sentencepiece`,
Torch, Transformers и Diffusers остаются частью проверяемого общего lock.

## Добавление следующей модели

1. Добавить `ImageEditProfile` и перечислить все пользовательские и runtime
   параметры без скрытых defaults.
2. Добавить ветку загрузки/вызова provider в worker. Editor/document/native UI
   менять не требуется.
3. Добавить fake-worker contract test, persistence test и opt-in GPU smoke.
4. Если checkpoint format отличается от Diffusers directory, реализовать
   изолированный loader в worker; не протаскивать format-specific поля в
   document/controller/UI.

## GPU smoke

Opt-in проверка не входит в обычный pytest и не скачивает модели неявно, если
передан `--local-files-only`:

```bash
./venv/bin/python scripts/smoke-image-edit.py \
  --profile flux2-klein-4b \
  --input input.png --output /tmp/flux-edit.png \
  --prompt "change only the car body" \
  --model /path/to/diffusers-pipeline --local-files-only
```

Дополнительные LoRA для smoke задаются повторяемым аргументом:

```bash
--lora /path/to/first.safetensors 0.8 \
--lora org/second-lora 0.55
```

Локальный SenseNova Q8 smoke:

```bash
./venv/bin/python scripts/smoke-image-edit.py \
  --profile sensenova-u1.5-8b-mot-preview \
  --worker-python ./.venv-workers/bin/python \
  --input input.png --output /tmp/sensenova-edit.png \
  --prompt "repaint only the car body in deep crimson red" \
  --steps 8 --seed 42015 --target-megapixels 1
```

Без `--model` используется upstream model ID профиля. В таком режиме Diffusers
может загрузить отсутствующий pipeline в Hugging Face cache; для Qwen и FLUX
это десятки гигабайт и такой запуск должен быть осознанным.
