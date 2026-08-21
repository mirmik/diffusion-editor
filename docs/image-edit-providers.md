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
dtype, device, offload, VAE tiling или LoRA вызывает reload.

## Встроенные профили

- `qwen-image-edit-2511` — `QwenImageEditPlusPipeline`, основной профиль.
  По умолчанию используются 4 шага при найденной Lightning LoRA, иначе 40.
- `flux2-klein-4b` — `Flux2KleinPipeline`, быстрый distilled профиль с четырьмя
  шагами.
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

Для локального Qwen по умолчанию включён `Component CPU offload`: text encoder
и transformer последовательно занимают GPU, а не находятся там одновременно.
Это соответствует автоматической перестановке компонентов в проверенном
ComfyUI workflow, но не включает послойную выгрузку transformer.

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

Без `--model` используется upstream model ID профиля. В таком режиме Diffusers
может загрузить отсутствующий pipeline в Hugging Face cache; для Qwen и FLUX
это десятки гигабайт и такой запуск должен быть осознанным.
