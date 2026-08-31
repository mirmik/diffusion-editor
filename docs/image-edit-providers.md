# Image Edit providers

AI Edit — один доменный инструмент и один generation lifecycle для моделей,
которые принимают исходное изображение и текстовую инструкцию. Модель не
кодируется в типе слоя, controller или native UI.

Генерация без исходного изображения намеренно вынесена в отдельный
[`TextToImageTool`](text-to-image.md): её область задаётся размером и позицией
целевого слоя, а не source patch или mask.

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
- `qwen-image-edit-rapid-aio-v23` — отдельный быстрый профиль Rapid AIO NSFW
  v23. Ускоряющая и стилистические LoRA уже слиты в transformer, поэтому
  профиль использует 4 шага, `true_cfg_scale = 1`, `guidance_scale = 1` и не
  добавляет Lightning повторно.
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

Qwen принимает независимые override'ы transformer и text encoder. Для encoder
в панели есть выбор `Upstream from Model`, `Standard scaled FP8`, `Heretic
BF16`, `Huihui Abliterated BF16` и `Custom file / directory`. Поэтому один
encoder можно сочетать с обычным 2511 transformer, Rapid AIO или другим
совместимым checkpoint'ом.
Если установленные ComfyUI scaled-FP8 файлы найдены, стандартный FP8 encoder
остаётся default для совместимости. Его Linear-веса хранятся в FP8 и
деквантизуются послойно для BF16-вычисления; precision-sensitive BF16/F32
параметры сохраняют исходный dtype. Upstream `Model` остаётся источником
processor, tokenizer, scheduler, VAE и конфигурации тех компонентов, для
которых override не выбран.

`Heretic BF16` автоматически ищет Transformers directory
`~/soft/ComfyUI/models/text_encoders/qwen-image-2512-heretic`, скачанный из
[`catplusplus/Qwen-Image-2512-Heretic`](https://huggingface.co/catplusplus/Qwen-Image-2512-Heretic).
Путь можно заменить через `DIFFUSION_EDITOR_QWEN_HERETIC_TEXT_ENCODER`.
Это модифицированный Qwen2.5-VL conditioning encoder, а не diffusion model и
не LoRA: он занимает место стандартного `text_encoder` и не мешает применять
transformer LoRA. Выбранные variant и фактический source входят в model
identity/provenance. `Custom` принимает scaled-FP8 файл, локальную Transformers
directory или Hugging Face repository ID.

`Huihui Abliterated BF16` ищется в
`~/soft/ComfyUI/models/text_encoders/qwen2.5-vl-7b-huihui-abliterated` и может
быть переопределён через `DIFFUSION_EDITOR_QWEN_HUIHUI_TEXT_ENCODER`. Источник —
[`huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated`](https://huggingface.co/huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated).
В отличие от Heretic, это аблитерированный upstream Qwen2.5-VL Instruct, а не
производная от text encoder Qwen Image 2512. Архитектура совместима с runtime;
качество conditioning и влияние аблитерации следует сравнивать на одинаковых
seed и prompt.

Rapid AIO v23 автоматически ищется в
`~/soft/ComfyUI/models/diffusion_models/qwen-image-edit-rapid-aio-v23`.
Путь можно заменить через
`DIFFUSION_EDITOR_QWEN_IMAGE_EDIT_RAPID_AIO_V23_TRANSFORMER`. Это шардированная
Diffusers-версия transformer из
[`prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23`](https://huggingface.co/prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23),
полученная из
[`Phr00t/Qwen-Image-Edit-Rapid-AIO`](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO).
Все её параметры хранятся в FP8; worker поднимает только исполняемый слой до
выбранного compute dtype и после forward возвращает его в FP8. Base pipeline,
processor, VAE и scheduler берутся из `Qwen/Qwen-Image-Edit-2511`; text encoder
выбирается независимо общим selector'ом.

При `true_cfg_scale <= 1` negative conditioning не вычисляется, поэтому
`Negative prompt` в этом режиме намеренно не влияет на результат. Для его
включения нужно установить `True CFG scale` больше `1`, однако это уже не
reference-режим Rapid AIO и требует больше памяти.

Multiple Angles автоматически ищется в
`~/soft/ComfyUI/models/loras/qwen-image-edit-2511-multiple-angles-lora.safetensors`.
Другой default задаётся через
`DIFFUSION_EDITOR_QWEN_MULTIPLE_ANGLES_LORA`. Prompt начинается с `<sks>` и
затем задаёт азимут, высоту камеры и дистанцию именно в таком порядке, например
`<sks> front-left quarter view elevated shot medium shot`. Multiple Angles не
является отдельным профилем: адаптер явно добавляется в стек основного
`qwen-image-edit-2511`, обычно с весом `1.0`.

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

Rapid AIO v23 smoke (веса должны быть заранее скачаны в default directory):

```bash
./venv/bin/python scripts/smoke-image-edit.py \
  --profile qwen-image-edit-rapid-aio-v23 \
  --worker-python ./.venv-workers/bin/python \
  --input input.png --output /tmp/qwen-rapid-v23-edit.png \
  --prompt "change only the jacket to deep crimson red" \
  --steps 4 --seed 42015 --width 1024 --height 1024 \
  --local-files-only
```

Без `--model` используется upstream model ID профиля. В таком режиме Diffusers
может загрузить отсутствующий pipeline в Hugging Face cache; для Qwen и FLUX
это десятки гигабайт и такой запуск должен быть осознанным.
