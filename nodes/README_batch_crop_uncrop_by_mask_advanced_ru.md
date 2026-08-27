# Batch Image Crop / Uncrop By Mask Advanced — документация

Ноды входят в [ComfyUI-StDismas](https://github.com/svyatojdismas/ComfyUI-StDismas) и предназначены для покадрового кадрирования изображений и видео с последующей точной вставкой обработанного crop обратно в исходный кадр.

<p align="center"><a href="README_batch_crop_uncrop_by_mask_advanced_en.md">English</a> | <a href="README_batch_crop_uncrop_by_mask_advanced_ru.md">Russian</a></p>

## Цели и возможности

Связка решает две основные задачи:

1. **Batch Image Crop By Mask Advanced (StDismas)** находит объект по маске или лицу, строит стабильную траекторию кадрирования и формирует batch crop одинакового разрешения.
2. **Batch Image Uncrop By Mask Advanced (StDismas)** возвращает обработанный crop в исходное изображение или видео и управляет областью, мягкостью и цветом вставки.

Основные возможности:

- работа с обычной маской любого объекта;
- автоматическое обнаружение и трекинг лица;
- удержание выбранной личности по `identity_reference` в сценах с несколькими людьми;
- интерполяция кадров, в которых маска или лицо временно отсутствуют;
- независимое сглаживание положения и масштаба crop;
- фиксированное, пользовательское или автоматически рассчитанное разрешение;
- переиспользование одной траектории через `pipe`;
- точный affine-uncrop через `crop_metadata`;
- совместимый прямоугольный uncrop через `bboxes` типа `BOUNDING_BOX`;
- вставка по маске объекта, по прямоугольной маске crop canvas или по всему crop;
- dilation, feathering, цветовое согласование и управление кадрами без детекции;
- обработка batch частями для ограничения расхода VRAM/RAM.

Типичный workflow:

```text
images/video ──► Crop ──► обработка cropped_images ──► Uncrop ──► full-frame result
     │             │                                      ▲
     └─────────────┴── base_images + crop_metadata ───────┘
```

Для обычной связки Crop → Uncrop рекомендуется использовать `crop_metadata`: она хранит дробные affine-координаты и точнее возвращает crop на место. Выход `bboxes` нужен для совместимости с другими нодами и для более простого прямоугольного способа вставки.

| Способ позиционирования | Точность | Что требуется | Когда использовать |
|---|---:|---|---|
| `crop_metadata` | максимальная | metadata из Crop; `base_images` желателен | основной workflow внутри ComfyUI-StDismas |
| `bboxes` | целые пиксели | `BOUNDING_BOX` и обязательно `base_images` | совместимость с другими crop/uncrop-нодами |

---

# Batch Image Crop By Mask Advanced (StDismas)

## Как работает Crop

Нода выполняет следующие этапы:

1. На каждом кадре получает bbox объекта из `crop_mask` либо от face detector.
2. Заполняет пропуски интерполяцией между соседними валидными кадрами.
3. Добавляет `margin_scale`, применяет `size_metric`, offsets и выбранное соотношение сторон.
4. Независимо сглаживает центр и размер окна.
5. Выбирает итоговое разрешение crop canvas.
6. Рассчитывает affine transform для каждого кадра.
7. Одним и тем же transform кадрирует изображение, главную и дополнительные маски.

Пустой кадр маски между двумя валидными кадрами не приводит к скачку в центр изображения: положение и размер восстанавливаются по предыдущему и следующему известному bbox. Для пустых кадров в начале или конце batch используется ближайшее известное значение.

## Основные входы

| Вход | Тип | Назначение |
|---|---|---|
| `images` | `IMAGE` | Исходный batch изображений или кадров видео. |
| `crop_mask` | `MASK` | Маска, по которой рассчитывается crop при `tracking_mode = mask`. Одна маска может быть автоматически размножена на весь batch. |
| `masks` | `MASK` | Необязательная дополнительная маска. Кадрируется тем же transform, но не влияет на геометрию crop. |
| `pipe` | `CROP_PIPE` | Параметры и готовая траектория из другой Crop-ноды. |
| `identity_reference` | `IMAGE` | Необязательный референс личности для face tracking. |

## Выбор источника траектории

### `tracking_mode = mask`

Это универсальный режим для человека, лица, предмета, области inpaint или любой другой маски.

- `crop_mask` обязателен;
- bbox рассчитывается по ненулевой области маски;
- пустые кадры интерполируются;
- дополнительный вход `masks` не меняет траекторию.

### `tracking_mode = face_detection`

Нода использует модель Ultralytics для обнаружения лица и строит траекторию по найденным face bbox. В этом режиме `crop_mask` для расчёта геометрии не нужен, а выход `cropped_masks` создаётся как прямоугольная маска области лица.

Face-параметры отображаются в интерфейсе только при выборе `face_detection`.

| Параметр | Значение по умолчанию | Назначение |
|---|---:|---|
| `face_detector` | первая доступная модель | Детектор из `models/ultralytics/bbox` или другого зарегистрированного Ultralytics-каталога. |
| `face_confidence` | `0.35` | Минимальная уверенность детектора. Меньше — больше найденных лиц и риск ложных детекций. |
| `face_select` | `largest` | Начальный выбор: самое крупное или наиболее центральное лицо. |
| `identity_track` | `true` | Использовать continuity и при необходимости InsightFace для удержания личности. |
| `identity_threshold` | `0.28` | Минимальное сходство с identity anchor. Большее значение делает выбор строже. |
| `identity_pack` | `buffalo_l` | Пакет InsightFace; `buffalo_s` требует меньше памяти. |
| `face_model_device` | `cpu` | Устройство для optional face-моделей: `cpu`, `auto` или `cuda`. |
| `keep_face_models_loaded` | `false` | Оставлять face-модели в памяти после расчёта. |
| `fallback_detector` | `none` | Необязательный person/body detector для кадров, где лицо пропало. |
| `fallback_head_frac` | `0.5` | Оценка вертикального положения головы внутри person bbox. |

### Identity tracking

Если подключён `identity_reference`, InsightFace извлекает embedding лица из первого изображения reference batch и использует его как identity anchor. Если reference не подключён, но в видео встречается несколько людей, нода пытается построить anchor по однозначным кадрам самого клипа.

В обычных кадрах сначала используется пространственная непрерывность: положение, размер и пересечение с последним выбранным лицом. InsightFace подключается для неоднозначных кадров, где рядом находятся несколько подходящих лиц. Такой подход уменьшает число тяжёлых identity-сравнений.

Если лицо временно не найдено:

- `fallback_detector` может оценить положение головы по bbox человека;
- оставшиеся пропуски заполняются интерполяцией;
- в `crop_metadata` сохраняется, была ли исходная face detection валидной;
- Uncrop затем может пропустить, плавно ослабить или всё равно скомпозить такой кадр.

Отдельная благодарность проекту [ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine) за открытый пример качественного face crop/refine workflow и полезные идеи по организации обратной вставки. В ComfyUI-StDismas face/identity tracking встроен в ту же универсальную Crop-ноду, что и режим обычной маски.

### Зависимости face tracking

Face-функции загружаются лениво и не используются при `tracking_mode = mask`. Для `face_detection` нужны:

- `ultralytics`;
- `insightface` для identity tracking;
- ровно один вариант ONNX Runtime: `onnxruntime` или `onnxruntime-gpu`;
- face detector, например `face_yolov8m.pt`, в каталоге `ComfyUI/models/ultralytics/bbox`.

Актуальный список Python-зависимостей находится в корневом `requirements.txt`. Не устанавливайте CPU- и GPU-варианты ONNX Runtime одновременно. Для минимального расхода VRAM используйте `face_model_device = cpu`, `identity_pack = buffalo_s` и `keep_face_models_loaded = false`.

## Геометрия и разрешение Crop

### Соотношение сторон

`aspect_ratio` поддерживает:

```text
1:1, 16:9, 9:16, 4:3, 3:4, 4:5, 5:4,
2:3, 3:2, 21:9, 9:21
```

Для произвольного соотношения включите `use_custom_aspect_ratio` и задайте `custom_aspect_ratio` в формате `W:H`, например `2.39:1`. Поле custom ratio показывается только при включённом переключателе.

### Режимы разрешения

| Параметр | По умолчанию | Поведение |
|---|---:|---|
| `resolution_mode` | `manual` | Способ выбора размера crop canvas. |
| `output_resolution_side` | `1024` | Целое число пикселей для выбранной стороны в ручном режиме. |
| `use_long_side` | `true` | `true` — значение относится к длинной стороне; `false` — к короткой. |
| `use_custom_resolution` | `false` | Использовать точные `width` и `height`. |
| `width` | `1024` | Пользовательская ширина; видна только при `use_custom_resolution = true`. |
| `height` | `576` | Пользовательская высота; видна только при `use_custom_resolution = true`. |
| `auto_resolution_cap` | `768` | Максимальная длинная сторона для `auto_capped`. |
| `divisible_by` | `2` | Делает обе стороны кратными указанному числу. |

Режимы `resolution_mode`:

- `manual` — использует `output_resolution_side` либо пользовательские `width`/`height`;
- `auto_no_downscale` — выбирает единый canvas по крупнейшему source crop во всём batch и округляет размеры минимум до кратности 32;
- `auto_capped` — работает так же, но ограничивает длинную сторону значением `auto_resolution_cap`.

Автоматические режимы сохраняют единый размер всех кадров batch. Фактический результат можно проверить через `canvas_width`, `canvas_height` и `report`.

### Размер объекта внутри crop

| Параметр | По умолчанию | Назначение |
|---|---:|---|
| `margin_scale` | `2.0` | Увеличивает исходный bbox до расчёта окна. Значения меньше `1.0` фактически работают как `1.0`. |
| `size_metric` | `bbox_fit` | Определяет, по какой размерности bbox строить crop. |
| `min_zoom` | `0.25` | Минимальный affine scale. |
| `max_zoom` | `6.0` | Максимальный affine scale. |
| `fit_frame_bounds` | `true` | Удерживает окно внутри исходного кадра, при необходимости изменяя центр и масштаб. |

Варианты `size_metric`:

- `bbox_fit` — универсально вписывает весь bbox в выбранный aspect ratio;
- `height` — ориентируется на высоту; обычно стабильнее для лица в профиль;
- `width` — ориентируется на ширину;
- `max_dimension` — использует наибольшую сторону bbox как квадратную базу;
- `area_sqrt` — использует квадратный корень из площади bbox.

`offset_x` и `offset_y` сдвигают центр crop в пикселях исходного изображения. Положительный `offset_x` двигает окно вправо, положительный `offset_y` — вниз.

## Сглаживание траектории

Положение и масштаб настраиваются независимо:

| Параметр | По умолчанию | Назначение |
|---|---:|---|
| `smooth_center` | `true` | Сглаживать движение центра. |
| `center_smooth_window` | `21` | Окно фильтра центра. |
| `center_smoothing_strength` | `0.25` | Скорость реакции центра на отфильтрованную траекторию. |
| `smooth_zoom` | `true` | Сглаживать размер окна и zoom. |
| `size_smooth_window` | `51` | Независимое окно фильтра размера. |
| `zoom_smoothing_strength` | `0.25` | Скорость реакции масштаба. |
| `smoothing_method` | `gaussian` | Метод фильтрации центра и размера. |

### Что делают `center_smooth_window` и `size_smooth_window`

Это не размер crop и не величина смещения. Это **число кадров, из которых строится сглаженное значение**.

- `center_smooth_window` влияет только на координаты центра окна: движение crop влево, вправо, вверх и вниз;
- `size_smooth_window` влияет только на размеры исходного окна, а значит на zoom: насколько близко или далеко объект оказывается в crop.

Например, при `center_smooth_window = 21` для центрального кадра нода учитывает приблизительно 10 кадров до него, текущий кадр и 10 кадров после него. Поэтому мелкий jitter маски или face detector усредняется, а плавное движение остаётся траекторией. При `size_smooth_window = 51` аналогично стабилизируется размер окна на более длинном отрезке, чтобы crop меньше «дышал» от небольших изменений bbox.

Чем больше окно, тем плавнее результат, но тем сильнее нода сглаживает резкие реальные изменения движения или масштаба. Рекомендуемый порядок настройки:

1. Сначала подберите `center_smooth_window` для плавности перемещения.
2. Затем независимо подберите `size_smooth_window`, если меняется zoom.
3. После этого настройте соответствующий `*_smoothing_strength`, чтобы задать скорость реакции.

Практические ориентиры:

| Сцена | `center_smooth_window` | `size_smooth_window` |
|---|---:|---:|
| Быстрое движение | `5–11` | `9–21` |
| Обычный разговорный ролик | `15–31` | `31–61` |
| Статичный или медленный кадр | `31–61` | `61–121` |

Окно применяется только к `gaussian`, `savgol` и `moving_average`. Для `ema` эти параметры не используются, а при `smoothing_method = none` сглаживание отсутствует. Нода ограничивает окно длиной batch и приводит его к нечётному значению, поэтому в коротком видео фактическое окно может быть меньше указанного.

Методы:

- `gaussian` — плавный симметричный фильтр, хороший основной вариант;
- `savgol` — лучше сохраняет форму движения; при недоступном SciPy автоматически используется Gaussian;
- `moving_average` — простое усреднение;
- `ema` — причинное экспоненциальное следование без оконного фильтра;
- `none` — полностью отключает сглаживание.

Для strength-параметров:

- `0` фиксирует траекторию на первом значении;
- малые значения дают больше инерции;
- `1` следует результату выбранного фильтра без дополнительного запаздывания.

Слишком большое окно может сделать движение очень плавным, но запаздывающим. Для быстрого объекта сначала уменьшайте окно, затем повышайте strength.

## Качество и производительность

### `interpolation`

- `bilinear` — быстрее и обычно достаточно для видео;
- `bicubic` — мягче при заметном масштабировании, но тяжелее.

Маски всегда семплируются через nearest interpolation, чтобы не создавать промежуточные значения на этапе Crop.

### `crop_chunk_size`

Значение по умолчанию — `128`. Это максимальное число кадров, семплируемых одной группой. Уменьшайте его при нехватке VRAM/RAM; увеличивайте, если памяти достаточно и нужен больший throughput.

Геометрия и resampling рассчитываются в FP32 даже для FP16/BF16-входов, после чего результат возвращается в исходный dtype.

## Переиспользование траектории через `pipe`

`pipe` переносит параметры и `crop_metadata` из другой Crop-ноды. Это позволяет одинаково кадрировать изображение, control image, depth, normals или другие связанные batch.

При наличии совместимой metadata нода переиспользует готовые affine transforms. Один кадр pipe может быть размножен на batch; в остальных случаях число кадров должно совпадать. Если разрешение изменилось пропорционально при том же aspect ratio, геометрия масштабируется. При другом aspect ratio точное совпадение невозможно и нода выдаёт ошибку.

`interpolation` и `crop_chunk_size` остаются локальными настройками принимающей ноды; остальные crop-параметры могут быть переопределены значениями из pipe.

## Выходы Crop

| Выход | Тип | Содержимое |
|---|---|---|
| `cropped_images` | `IMAGE` | Кадрированный batch изображений. |
| `cropped_masks` | `MASK` | Главная маска в crop-пространстве; в face mode — прямоугольная маска face bbox. |
| `masks` | `MASK` | Дополнительная маска после того же transform; без входа `masks` повторяет `cropped_masks`. |
| `visualize` | `IMAGE` | Исходные кадры с красной рамкой crop. Рассчитывается только если выход подключён. |
| `crop_metadata` | `BBOXES` | Точные per-frame affine transforms, размеры, валидность tracking и служебная статистика. |
| `bboxes` | `BOUNDING_BOX` | Округлённые прямоугольники итогового crop window. |
| `pipe` | `CROP_PIPE` | Параметры и metadata для другой Crop-ноды. |
| `report` | `STRING` | Режим, число валидных кадров, canvas, magnification, jitter и предупреждения face tracking. |
| `canvas_width` | `INT` | Итоговая ширина crop canvas. |
| `canvas_height` | `INT` | Итоговая высота crop canvas. |

Формат `bboxes` совместим с подходом из [MaskVidExperiments](https://github.com/drozbay/MaskVidExperiments):

```text
[
  [{"x": 100, "y": 50, "width": 512, "height": 512}],
  [{"x": 104, "y": 52, "width": 512, "height": 512}]
]
```

Это bbox итогового affine crop window, а не сырой bbox маски. Координаты округлены до целых пикселей, поэтому для точного обратного сшивания внутри этой библиотеки предпочтительнее `crop_metadata`.

## Практические настройки Crop

Стабильный face crop:

```text
tracking_mode = face_detection
size_metric = height
margin_scale = 1.5–2.5
smoothing_method = gaussian
center_smooth_window = 15–31
size_smooth_window = 31–61
fit_frame_bounds = true
```

Быстрое движение:

```text
center_smooth_window = 5–11
center_smoothing_strength = 0.5–0.9
```

Минимум пульсации zoom:

```text
smooth_zoom = true
size_smooth_window = 41–81
zoom_smoothing_strength = 0.15–0.4
```

---

# Batch Image Uncrop By Mask Advanced (StDismas)

## Как работает Uncrop

Uncrop получает обработанный `cropped_images`, восстанавливает его положение в исходном кадре и смешивает с `base_images` выбранным способом.

Есть два пути позиционирования:

1. **Affine path через `crop_metadata`** — использует точный transform Crop-ноды, поддерживает дробные координаты, tracking validity, color matching и автоматическое ограничение chunk по памяти.
2. **BBox path через `bboxes`** — масштабирует crop до прямоугольника `{x, y, width, height}`. Этот путь проще и совместим с другими нодами, но теряет subpixel-точность.

Если подключены оба входа, приоритет имеет `crop_metadata`.

## Входы Uncrop

| Вход | Тип | Назначение |
|---|---|---|
| `cropped_images` | `IMAGE` | Обработанный batch crop, который нужно вернуть в полный кадр. |
| `base_images` | `IMAGE` | Фоновый batch для compositing. Обязателен для bbox path. |
| `original_images` | `IMAGE` | Legacy alias для `base_images`. |
| `crop_metadata` | `BBOXES` | Точная metadata из Crop-ноды. |
| `bboxes` | `BOUNDING_BOX` | Прямоугольники в формате `[[{x, y, width, height}], ...]`. |
| `crop_masks` | `MASK` | Alpha source для mask-based stitching. |

При affine path `base_images` можно не подключать: тогда создаётся чёрный canvas размера `orig_size` из metadata. Для обычного workflow подключайте исходные кадры как `base_images`.

## Режимы вставки

По умолчанию используются `mode = overlay_by_mask`, `blend = 1.0` и `use_crop_canvas_mask = true`.

### `mode = overlay_by_mask`

Использует alpha-mask. Конкретный источник alpha выбирает `use_crop_canvas_mask`:

- `false` — маска объекта из `crop_masks`, расширенная и сглаженная;
- `true` — отдельная прямоугольная маска crop canvas с индивидуальными inset/fade по сторонам.

### `mode = overlay_full`

Заменяет весь crop window. `crop_masks`, `mask_expand_px`, `feather_radius` и настройки прямоугольной маски в этом режиме не определяют форму alpha.

Режим удобен, когда обработанный crop должен полностью заменить прямоугольный участок, но может показать швы, если crop отличается по цвету, шуму или детализации от base image.

### `blend`

Общая сила вставки от `0.0` до `1.0`:

- `0.0` — оставить base image;
- `0.5` — применить половину alpha;
- `1.0` — полностью применить выбранную вставку.

## Вставка по `crop_masks`

Установите:

```text
mode = overlay_by_mask
use_crop_canvas_mask = false
```

Порядок обработки alpha:

```text
crop_masks → dilation(mask_expand_px) → перенос в исходный кадр → Gaussian feather → blend
```

| Параметр | По умолчанию | Назначение |
|---|---:|---|
| `mask_expand_px` | `16` | Расширяет плотную область маски до размытия. Измеряется в пикселях crop canvas. |
| `feather_radius` | `16` | Размывает уже расширенный край. |
| `border_blending` | `0.25` | Legacy fallback: используется для вычисления feather, если `feather_radius = 0`. |

Dilation и feathering — не одно и то же. Dilation увеличивает область, внутри которой новый crop заменяет base image; feathering создаёт мягкий переход по новому краю. Такая последовательность помогает убрать ореол исходного объекта, не делая всю область вставки полупрозрачной.

Если нужен полностью жёсткий край, установите одновременно:

```text
mask_expand_px = 0
feather_radius = 0
border_blending = 0
```

## Прямоугольная маска crop canvas

Установите:

```text
mode = overlay_by_mask
use_crop_canvas_mask = true
```

Этот режим не использует форму объекта из `crop_masks`. Он создаёт прямоугольную alpha-mask внутри crop и позволяет отдельно настроить четыре стороны.

| Параметры | По умолчанию | Назначение |
|---|---:|---|
| `square_mask_inset_left_px` | `8` | Сдвиг активной области от левого края. |
| `square_mask_inset_right_px` | `8` | Сдвиг от правого края. |
| `square_mask_inset_top_px` | `8` | Сдвиг от верхнего края. |
| `square_mask_inset_bottom_px` | `8` | Сдвиг от нижнего края. |
| `square_mask_fade_left_px` | `16` | Ширина мягкого перехода слева. |
| `square_mask_fade_right_px` | `16` | Ширина перехода справа. |
| `square_mask_fade_top_px` | `16` | Ширина перехода сверху. |
| `square_mask_fade_bottom_px` | `16` | Ширина перехода снизу. |

`inset` определяет, где расположена полностью активная прямоугольная область. `fade` определяет ширину перехода от alpha `0` к `1`. Это независимые параметры.

Несмотря на историческое имя `square_mask_*`, маска работает с crop canvas любого aspect ratio.

### `square_mask_units`

- `crop_pixels` — inset/fade задаются в пикселях crop canvas; их физическая ширина в исходном кадре меняется вместе с zoom;
- `source_pixels` — нода компенсирует affine scale, чтобы видимая ширина перехода оставалась примерно постоянной в пикселях исходного кадра.

Компенсация `source_pixels` применяется в точном affine path. В простом bbox path настройки используются в пикселях области вставки.

## Цветовое согласование

`color_match_mode` доступен в affine path через `crop_metadata`:

| Режим | Поведение |
|---|---|
| `off` | Не изменять цвет crop. |
| `mean` | Согласовать средние значения каналов. |
| `mean_std` | Согласовать среднее и контраст каналов. |
| `luminance` | Согласовать яркость, сохраняя цветовой характер лучше, чем полное поканальное matching. |

`color_match_strength` задаёт силу коррекции от `0.0` до `1.0`. Слишком сильное согласование может менять художественный цвет обработанного crop, поэтому начинать лучше с небольших значений.

## Кадры без валидной детекции

При face tracking `crop_metadata` хранит флаг `valid` отдельно для каждого кадра. Параметр `undetected_frames` определяет только обратную вставку; кадры Crop всё равно остаются доступны генератору.

- `fade_out` — плавно уменьшает вклад вставки рядом с пропусками;
- `skip` — не вставляет кадры без валидной face detection;
- `composite_anyway` — вставляет все кадры по интерполированной траектории.

`dropout_fade_window` по умолчанию равен `9` и управляет временной шириной перехода для `fade_out`.

## Производительность Uncrop

| Параметр | По умолчанию | Назначение |
|---|---:|---|
| `uncrop_chunk_size` | `4` | Максимальное число full-frame warp за один проход. |
| `uncrop_memory_limit_mb` | `512` | Бюджет временных FP32-тензоров; при необходимости автоматически уменьшает фактический chunk. |
| `crop_rescale` | `1.0` | Масштаб patch в legacy/bbox path. На актуальный affine path не влияет. |

Full-frame affine warp требует значительно больше памяти, чем Crop. При высоком разрешении уменьшайте `uncrop_chunk_size` или `uncrop_memory_limit_mb`.

## Выход Uncrop

| Выход | Тип | Содержимое |
|---|---|---|
| `images` | `IMAGE` | Итоговый full-frame batch после compositing. |

## Практические настройки Uncrop

Если по краю объекта остаётся ореол исходного изображения:

```text
mode = overlay_by_mask
use_crop_canvas_mask = false
mask_expand_px = 8–32
feather_radius = 8–24
```

Если новая генерация сильно изменила форму объекта и старая маска уже не подходит:

```text
mode = overlay_by_mask
use_crop_canvas_mask = true
square_mask_inset_* = 4–16
square_mask_fade_* = 16–64
```

Если видны вертикальные швы, увеличьте `square_mask_fade_left_px` и `square_mask_fade_right_px`. Если шов заметен снизу — увеличьте `square_mask_fade_bottom_px`. Сторону, которую нельзя изменять, можно сильнее отодвинуть соответствующим inset.

---

# Рекомендуемые подключения

## Основной точный workflow

```text
Load Images / Video
        │
        ├──────────────► base_images (Uncrop)
        │
        ▼
Batch Image Crop By Mask Advanced
        │ cropped_images
        ▼
Generation / Inpaint / Detailer
        │
        └──────────────► cropped_images (Uncrop)

Crop.crop_metadata ───► Uncrop.crop_metadata
Crop.cropped_masks ───► Uncrop.crop_masks
```

## BOUNDING_BOX workflow

```text
Crop.bboxes ──────────► Uncrop.bboxes
original frames ──────► Uncrop.base_images
processed crop ───────► Uncrop.cropped_images
```

Этот вариант совместим с нодами, которые используют `BOUNDING_BOX`, но для Crop и Uncrop из одного пакета точнее первый workflow.

---

# Частые проблемы

## Crop прыгает по кадру

- включите `smooth_center`;
- используйте `gaussian`;
- увеличьте `center_smooth_window`;
- уменьшите `center_smoothing_strength`;
- для нескольких лиц подключите `identity_reference`.

## Crop отстаёт от быстрого объекта

- уменьшите `center_smooth_window`;
- увеличьте `center_smoothing_strength`;
- для минимальной задержки используйте `ema` или `none`.

## Crop пульсирует по масштабу

- включите `smooth_zoom`;
- увеличьте `size_smooth_window`;
- уменьшите `zoom_smoothing_strength`;
- для лица попробуйте `size_metric = height`.

## Crop слишком приближается

- увеличьте `margin_scale`;
- уменьшите `max_zoom`;
- проверьте стабильность исходной маски или face bbox.

## Появляются чёрные поля по краям Crop

Включите `fit_frame_bounds = true`. Нода удержит affine window внутри исходного кадра, хотя композиция рядом с границей может немного измениться.

## Face detector не находит лицо

- проверьте, что detector находится в `models/ultralytics/bbox`;
- уменьшите `face_confidence`;
- проверьте выбранный `face_model_device`;
- для далёкого человека подключите подходящий `fallback_detector`;
- если лицо не является целью workflow, используйте `tracking_mode = mask`.

## Выбирается не тот человек

- подключите чёткий `identity_reference` с одним хорошо видимым лицом;
- оставьте `identity_track = true`;
- при ложных совпадениях увеличьте `identity_threshold`;
- проверьте `report`: строка face tracking показывает число identity/continuity решений и источник anchor.

## Uncrop оставляет контур старого объекта

При mask-based stitching увеличьте `mask_expand_px`, а затем подберите `feather_radius`. Одно только сильное размытие не заменяет расширение маски.

## Uncrop создаёт прямоугольные швы

- используйте `overlay_by_mask` вместо `overlay_full`;
- настройте fade отдельно для проблемной стороны;
- попробуйте `square_mask_units = source_pixels` при меняющемся zoom;
- включите мягкое `color_match_mode`;
- убедитесь, что processed crop не изменил фон слишком сильно.

## Mask-based Uncrop не совпадает с новым объектом

Если генерация изменила силуэт, старая `crop_masks` может стать неподходящей. Переключитесь на `use_crop_canvas_mask = true` либо подайте новую маску результата.

## Не хватает памяти

- уменьшите `crop_chunk_size` для Crop;
- уменьшите `uncrop_chunk_size` и `uncrop_memory_limit_mb` для Uncrop;
- используйте `face_model_device = cpu`;
- установите `identity_pack = buffalo_s`;
- оставьте `keep_face_models_loaded = false`.

---

# Совместимость сохранённых workflow

- Старые workflow с отдельной Crop By Mask Or Face нодой автоматически переводятся на универсальную Crop-ноду при загрузке.
- Старое имя `output_long_side` мигрирует в `output_resolution_side`.
- Старое имя `use_square_mask` мигрирует в `use_crop_canvas_mask`.
- Для новых workflow используйте актуальные имена из этой документации.

После обновления custom node перезапустите ComfyUI и обновите страницу браузера, чтобы frontend перечитал схему нод и динамические поля интерфейса.
