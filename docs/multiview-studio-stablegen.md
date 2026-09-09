# Ручные текстурные проходы StableGen

В нативной Multiview Studio добавлена панель **Texture · camera image and projection**.
Это отдельный ручной проход по схеме StableGen: SDXL inpainting,
Depth ControlNet и IPAdapter. Изображение генерирует отдельный процесс
Diffusers; ComfyUI и Blender не используются.

## Как пользоваться

Все действия выполняются в Multiview Studio; основной редактор не открывается.

1. Выбрать целую модель, установить ракурс и нажать **Edit current camera**.
   Камера и размеры полного кадра фиксируются для этого прохода.
2. Выбрать **SDXL checkpoint** из списка локальных моделей либо указать файл
   кнопкой **Choose SDXL checkpoint**. Depth ControlNet и IPAdapter продолжают
   работать с выбранным checkpoint. При смене checkpoint дополнительная
   Lightning LoRA отключается; её можно включить отдельным списком.
3. Нажать **Draw Patch** и протянуть прямоугольник на изображении. Зелёная
   рамка обозначает фрагмент генерации. **Full image Patch** убирает
   ограничение. Повторное нажатие **Paint mask** возвращает кисть.
4. Нарисовать маску изменений внутри Patch: левая кнопка добавляет,
   правая стирает. Настроить prompt, reference, denoise, ControlNet, IPAdapter,
   steps и CFG. Рабочее разрешение Patch: 512, 768 или 1024 по длинной стороне.
5. Нажать **Generate image in Patch**. RGB, depth и маска обрезаются одной
   рамкой и приводятся к одному рабочему размеру. Результат вклеивается
   обратно в полный кадр. Вне Patch и нулевой маски пиксели сохраняются.
   Каждая генерация использует исходный захваченный кадр, включая повторные
   попытки и смену seed. Предыдущий PNG-кандидат заменяется новым результатом.
6. Проверить картинку и отдельно нажать **Project image onto mesh**.
   Только эта кнопка выполняет UV-проекцию и показывает её на модели.
7. **Apply projected texture** принимает текстуру. **Discard / close pass**
   отменяет непринятый проход. **Undo / Redo texture pass** переключают
   принятые состояния. **Export GLB** сохраняет принятую модель с текстурами.

Checkpoint должен быть совместим с используемыми SDXL ControlNet/IPAdapter.
Список определяется по архитектуре в заголовке safetensors, без загрузки
весов. SD 1.5, FLUX и другие архитектуры этим worker не поддерживаются.
Например, DreamShaper XL Lightning можно выбрать с **No acceleration LoRA**;
при выборе обычного SDXL начальные параметры — 30 steps / CFG 5,
для Lightning — 8 / 1.5. **Use model sampling defaults** повторно выставляет
этот набор для уже выбранного checkpoint. Параметры можно менять вручную.

**Import edited image** остаётся способом загрузить готовый полный кадр.
Размер должен совпадать с захваченной камерой. **Show before / Show candidate**
сравнивают изображения и, после проекции, материалы в 3D. Изменение картинки
или маски сбрасывает старую проекцию; перед принятием её нужно повторить.

Артефакты лежат рядом с проектом в `texture-runs/stablegen-…`:
исходный GLB, камера, полный RGB/depth, маска, `patch.json`,
`generation-input.png`, `patch-rgb.png`, `patch-depth.png`,
`patch-mask.png`, `patch-result.png`, полный PNG-кандидат,
настройки генерации и диагностика UV-проекции. Отчёт `generation.json`
фиксирует checkpoint, использованную LoRA, рамку и рабочий размер Patch.
При сохранении проекта копируются артефакты его принятой истории.

## Запуск вычислений

Запустить Studio обычным способом:

```bash
./run-multiview-studio.sh
```

**Generate image** автоматически запускает одноразовый Python worker.
После генерации изображения процесс завершается и освобождает CUDA-память.
Проекция запускается отдельной кнопкой и отдельным процессом. Отмена завершает worker
через terminate с kill по таймауту и ожидает его выхода. Следующая операция
Pixal3D/TRELLIS запускается после завершения текущего задания. Серверы и порты
не нужны.

Worker использует Python из `DIFFUSION_EDITOR_STABLEGEN_PYTHON`, по умолчанию
`/home/mirmik/soft/TRELLIS.2/venv/bin/python`. Проверены diffusers 0.37.1,
transformers 4.57.3, accelerate 1.13.0, peft 0.17.1, torch/CUDA, safetensors,
Pillow и numpy. Для UV-проекции нужны также nvdiffrast, cumesh и trimesh.

Веса загружаются непосредственно из локальных файлов:

- `checkpoints/RealVisXL_V5.0_fp16.safetensors`;
- `loras/sdxl_lightning_8step_lora.safetensors`;
- `controlnet/controlnet_depth_sdxl.safetensors`;
- `ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors`;
- `clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`.

Корень весов задаёт `DIFFUSION_EDITOR_STABLEGEN_MODELS`. На текущей машине
по умолчанию это существующее хранилище `/home/mirmik/soft/ComfyUI/models`:
используются только файлы весов, код ComfyUI не импортируется и сервер
не запускается. В проектном JSON можно указать абсолютные пути к весам.

Конфигурации и токенизаторы SDXL берутся из локального Hugging Face cache
`stabilityai/stable-diffusion-xl-base-1.0`. Для другого расположения указать
`DIFFUSION_EDITOR_STABLEGEN_SDXL_CONFIG` — каталог snapshot с model_index,
конфигурациями компонентов и двумя токенизаторами. Генерация работает offline
и не скачивает недостающие файлы неявно.

Начальные параметры: 8 выполняемых шагов, CFG 1.5, denoise 0.55, Depth 0.75,
IPAdapter 0.65, длинная сторона снимка 768. Имена моделей и размер хранятся
в секции `stablegen` проектного JSON. Старое поле `comfy_url` игнорируется
при загрузке и удаляется при сохранении.

**Prediction: Auto** использует то же определение v_prediction по имени
checkpoint, что и основной редактор. Доступны явные epsilon/v_prediction.
**Sampler: Auto** выбирает Euler для дополнительной Lightning LoRA, иначе
DPM++ SDE Karras как в основном редакторе. Выбор можно переопределить.
Итоговый prediction mode и sampler записываются в generation.json.

**Reset image to captured view** возвращает предпросмотр к исходному кадру,
сохраняя Patch и маску. Генерация всегда начинает с исходного кадра независимо
от того, какой кандидат показан в предпросмотре.

Мягкая маска постепенно разрешает изменения по шагам; нулевые значения
остаются закрыты. Пиксели вне маски восстанавливаются из исходного RGB также
после VAE. Это новый sampler и реализация маски: пиксельного совпадения
с прежним Comfy-графом не предполагается. Отчёт `generation.json` сохраняет
backend, версии, пути весов, sampler, число шагов, время и пик VRAM.

Компоненты: [IPAdapter в Diffusers](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter),
[SDXL Lightning](https://huggingface.co/ByteDance/SDXL-Lightning).
Архитектуры загрузчиков соответствуют конфигурациям
[Depth ControlNet](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/blob/main/config.json)
и [CLIP ViT-H](https://huggingface.co/h94/IP-Adapter/blob/main/models/image_encoder/config.json).

## Границы первого варианта

Редактируется base color статического GLB с одним мешем/материалом.
Геометрия и остальные PBR-карты сохраняются. При отсутствии UV создаётся
атлас; при отсутствии материала используется нейтральная основа.
Vertex colors требуют предварительного запекания в текстуру.

RGB снимается без освещения и оверлеев, depth — из той же камеры.
Обратная проекция проверяет первый ray hit, маску и угол поверхности.
Цвет смешивается в линейном пространстве; texels вне разрешённой области
остаются побитно прежними. Перекрывающиеся UV texels и соседний пояс
защищены от записи; полностью неоднозначный атлас отклоняется.
`protected-uv-texels.png` показывает эту защиту, `allowed-texels.png` —
разрешённую область конкретного прохода.

Нейросеть может менять детали внутри маски — качество нужно оценивать
в предпросмотре перед принятием. Проверка полного покрытия Vaan ведётся
отдельно в Kanboard #2276.

## Проверенный случай SmoothYaoiBoys v30Vpred

На пользовательском Patch Vaan режим epsilon воспроизводил яркие цветные
артефакты. Для этого checkpoint нужен v_prediction. Проверенный вариант:
DPM++ SDE Karras, 30 steps, CFG 5, denoise 0.5, Depth 0.75, IPAdapter 0,
без Lightning LoRA. На том же исходнике IPAdapter 0.65 воспроизводил искажения
как с Depth, так и без него; это не доказательство несовместимости всех
v_prediction-моделей с IPAdapter.

После неудачных проходов сначала использовать **Reset image to captured view**,
затем **Use model sampling defaults**, выставить IPAdapter 0 и генерировать заново.
Проверки сохранены в `.local/stablegen-studio/vpred-*` и `epsilon-clean`.
