# Text to Image

Text to Image — отдельный provider-neutral инструмент для моделей, которые
создают изображение без исходных пикселей. Он не использует `DiffusionTool` и
не является режимом AI Edit: у запроса нет source image, patch или mask. Поддерживающие это профили
могут получать отдельный необязательный список визуальных референсов; пиксели целевого
слоя при этом не используются.

В Text to Image также доступны локальные **Qwen Image 2.1**, **Qwen Image 2.1
Turbo8** (community LoRA) и **Qwen Image 2.1 Turbo (Official)**. Они поддерживают
RGBA и до 10 упорядоченных референсов из слоёв и файлов. Настройка и проверка описаны в
[Qwen Image 2.1](qwen-image-2.1.md).

## Референсы

В профилях Qwen 2.1 блок **Reference images** позволяет добавлять изображения
кнопкой **Add Image...** и выбирать слои через **Add layer reference…**.
Можно смешивать файлы и слои; каждый новый референс добавляется в конец списка.
Стрелки меняют порядок, **×** удаляет отдельный элемент, **Clear All** очищает
список. Все эти действия поддерживают undo/redo.

Номера `<image1>`, `<image2>`, … соответствуют текущему порядку списка и меняются
при перестановке или удалении. В промпте задавайте роль каждого изображения,
например: «Используй палитру и стиль <image1>, персонажа из <image2>, композицию
из <image3>». Отдельного регулятора силы стиля нет.

Файлы встраиваются в `.deproj` с сохранением альфы; ссылки на слои, порядок и
идентификаторы элементов тоже сохраняются. Старые проекты с одним референсом
загружаются как список из одного элемента. Перед генерацией фиксируются пиксели
всех входов и их хеши; изменение любого референса или порядка отклоняет устаревший
результат. Замена списка не требует повторной загрузки весов модели.

## Геометрический контракт

`TextToImageTool` прикрепляется к обычному raster layer. Размер результата
всегда равен `layer.width × layer.height`; позиция `layer.x, layer.y` задаёт
область этого результата на холсте. Размер холста в запрос не передаётся.

При старте controller замораживает bounds слоя и полный локальный прямоугольник
`(0, 0, width, height)`. Изменение позиции, размера, пикселей, tool settings или
document revision отменяет pending job либо отклоняет поздний результат.
Успешный результат заменяет все пиксели слоя одной document-командой и поэтому
поддерживает undo/redo.

## Профили и worker

Исходные два независимых профиля одного provider `diffusers.qwen_image`:

- `qwen-image-2512` (профиль по умолчанию), model ID
  `Qwen/Qwen-Image-2512`;
- `qwen-image` — оригинальная модель августа 2025 года, model ID
  `Qwen/Qwen-Image`.

Оба используют `QwenImagePipeline`. Model ID профиля 2512 можно переопределить
переменной `DIFFUSION_EDITOR_QWEN_IMAGE_MODEL`, а оригинального —
`DIFFUSION_EDITOR_QWEN_IMAGE_BASE_MODEL`; то же самое доступно через поле
`Model / local directory`. Параметры и LoRA stack сохраняются отдельно для
каждого профиля и восстанавливаются при переключении.

`TextToImageRequest` содержит stable profile ID, effective parameters,
LoRA stack, точные `width`, `height` и упорядоченный `reference_images` (пустой по умолчанию). Изолированный ML worker обслуживает
операции `load_text_to_image` и `text_to_image`. Load-time identity включает
model, revision, dtype, device, offload, VAE tiling и LoRA; prompt и sampling
parameters не требуют reload.

По умолчанию Qwen Image 2512 использует локальный Comfy-style scaled-FP8
transformer, а оригинальный профиль — `qwen_image_fp8mixed.safetensors` из
[`Comfy-Org/Qwen-Image_ComfyUI`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI).
Оба используют общий с Qwen Image Edit scaled-FP8 text encoder. Наш loader
хранит Linear weights в FP8, а вычисляет в BF16; внешний `torchao` или ComfyUI
runtime не нужен. Профиль автоматически находит файлы в
`~/soft/ComfyUI/models/diffusion_models` и `text_encoders`. Переменные
`DIFFUSION_EDITOR_QWEN_IMAGE_TRANSFORMER` и
`DIFFUSION_EDITOR_QWEN_IMAGE_BASE_TRANSFORMER` задают transformer для 2512 и
оригинального профиля соответственно;
`DIFFUSION_EDITOR_QWEN_IMAGE_TEXT_ENCODER` задаёт общий text encoder.

Оригинальный профиль не подключает ускоряющую LoRA автоматически и стартует с
25 шагами и True CFG `3.0`. Это также подходящий профиль для LoRA, обученных
на исходной Qwen Image, а не на базе 2512.

Text encoder выбирается независимо от transformer: `Upstream from Model`,
`Standard scaled FP8`, `Heretic BF16`, `Huihui Abliterated BF16` или
`Custom file / directory`. Heretic
автоматически ищется в
`~/soft/ComfyUI/models/text_encoders/qwen-image-2512-heretic`; другой путь
задаётся `DIFFUSION_EDITOR_QWEN_HERETIC_TEXT_ENCODER`. Это полноценная BF16
Transformers directory из
[`catplusplus/Qwen-Image-2512-Heretic`](https://huggingface.co/catplusplus/Qwen-Image-2512-Heretic),
которая заменяет только Qwen2.5-VL conditioning encoder. Она совместима с FP8
Qwen Image 2512 transformer и не конфликтует с Lightning или другими
transformer LoRA. Выбранные variant и source сохраняются в model identity.

Huihui автоматически ищется в
`~/soft/ComfyUI/models/text_encoders/qwen2.5-vl-7b-huihui-abliterated`; другой
путь задаётся `DIFFUSION_EDITOR_QWEN_HUIHUI_TEXT_ENCODER`. Это BF16
[`huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated`](https://huggingface.co/huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated).
Он архитектурно совместим с Qwen Image conditioning, но получен из upstream
Qwen2.5-VL Instruct, поэтому не предполагается эквивалентным стандартному или
Heretic encoder по качеству.

При наличии
`~/soft/ComfyUI/models/loras/Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors`
профиль также включает `lightx2v/Qwen-Image-2512-Lightning` с весом `1.0`.
С ним sampling defaults меняются с 50 шагов и True CFG `4.0` на 4 шага и
True CFG `1.0`. Путь можно переопределить через
`DIFFUSION_EDITOR_QWEN_IMAGE_LORA`. LoRA остаётся в BF16 поверх scaled-FP8
base weights; worker явно приводит параметры adapter'а к compute dtype, чтобы
они не были ошибочно сохранены как неотмасштабированный FP8.

LoRA stack доступен в Text to Image panel так же, как в AI Edit: Lightning
можно отключить, удалить, заменить или дополнить adapter'ами из общего LoRA
catalog. Явно пустой stack сохраняется и не заполняется заново после reload.
Документы ранней версии Text to Image один раз принимают Lightning и новые
sampling defaults, только если в них оставались прежние значения 50/4.0.

`VRAM strategy = Component CPU offload` включён по умолчанию: крупнейший FP8
component помещается в 32 GiB вместе с активациями, а неиспользуемые components
остаются в RAM. `Group CPU offload` предназначен для диагностического BF16
fallback, `Sequential CPU offload` экономит ещё больше VRAM ценой существенного
замедления, а `Resident on device` не использует offload. Старое сохранённое
`cpu_offload=true` мигрирует в component mode, `false` — в resident mode.

Профиль, все значения параметров, LoRA stack и provenance сохраняются в
`.deproj`. Native-панель явно показывает итоговый размер слоя и не показывает
Mask Brush для Text to Image.

## Добавление provider

Следующая модель добавляется как `TextToImageProfile` и worker adapter с тем же
request/result contract. Тип document tool, controller и UI при этом менять не
нужно. Provider не должен самовольно менять размер итогового результата:
engine проверяет его перед передачей в document lifecycle. Qwen требует
стороны, кратные 16: worker округляет внутреннее inference-разрешение вверх и
после decode приводит изображение к точному размеру слоя. Оба внутренних
размера записываются в runtime provenance.
