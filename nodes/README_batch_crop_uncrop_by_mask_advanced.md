# Batch Image Crop / Uncrop By Mask Advanced — документация

Ноды предназначены для **кадрирования batch/video по маске** и последующей **обратной вставки cropped-результата в исходное изображение/видео**.

Репозиторий: https://github.com/svyatojdismas/ComfyUI-StDismas

## Новые режимы FaceRefine

Универсальный mask-пайплайн сохранен. Одна нода **Batch Image Crop By Mask Advanced** поддерживает оба режима: `crop_mask` обязателен только при `tracking_mode = mask`.
Сохранённые workflow со старой временной нодой **Batch Image Crop By Mask or Face Advanced** автоматически переходят на эту универсальную ноду при загрузке.

- `tracking_mode = mask` — обычная универсальная работа по любой маске;
- `tracking_mode = face_detection` — поиск и покадровый трекинг лица;
- `identity_reference` — необязательный референс личности;
- continuity выбирает лицо в обычных кадрах, InsightFace вызывается только при неоднозначном выборе;
- `fallback_detector` может оценить положение головы по bbox человека, если лицо временно пропало;
- `keep_face_models_loaded = false` освобождает detector/recognizer до генерации и является безопасным режимом для ограниченных VRAM/RAM.

Face-функции загружаются лениво и не нужны для `tracking_mode = mask`. Для них отдельно установите `requirements-face.txt`, один вариант ONNX Runtime и face detector (например `face_yolov8m.pt`) в `models/ultralytics/bbox`. Не устанавливайте одновременно CPU- и GPU-варианты ONNX Runtime. Для минимального расхода VRAM используйте `face_model_device = cpu`, `identity_pack = buffalo_s`, `keep_face_models_loaded = false`.

### Траектория и разрешение

- `smoothing_method`: `gaussian`, `savgol`, `moving_average`, прежний `ema` или `none`;
- `center_smooth_window` и `size_smooth_window` сглаживают положение и масштаб раздельно;
- прежние `center_smoothing_strength` и `zoom_smoothing_strength` управляют реакцией всех методов: `0` фиксирует центр/zoom, `1` полностью следует отфильтрованной траектории;
- интерфейс показывает окна только для оконных фильтров, а параметры face tracking — только при `tracking_mode = face_detection`;
- пустые кадры интерполируются между прошлым и будущим valid bbox;
- `size_metric = bbox_fit` остается универсальным, `height` рекомендуется для лица в профиль;
- `resolution_mode = manual` сохраняет ручное разрешение;
- `auto_no_downscale` выбирает единый canvas по крупнейшему source crop;
- `auto_capped` делает то же, но ограничивает длинную сторону через `auto_resolution_cap`;
- дополнительные выходы `report`, `canvas_width`, `canvas_height` показывают выбранное разрешение и magnification statistics.

### Улучшенный Uncrop

- `square_mask_units = source_pixels` сохраняет физическую ширину inset/fade при изменении zoom; `crop_pixels` сохраняет прежнее поведение;
- feather теперь Gaussian;
- `color_match_mode`: `off`, `mean`, `mean_std`, `luminance`; интенсивность задается `color_match_strength`;
- `undetected_frames`: `fade_out`, `skip`, `composite_anyway`;
- `uncrop_chunk_size` ограничивает размер GPU batch, а `uncrop_memory_limit_mb` автоматически уменьшает chunk, чтобы не переполнить VRAM/RAM.

---

# 1. Batch Image Crop By Mask Advanced

## Назначение

**Batch Image Crop By Mask Advanced** берет batch изображений или видео-кадров, анализирует маску `crop_mask`, находит область объекта по этой маске и вырезает crop вокруг нее.

Нода поддерживает:

- crop по маске;
- настройку aspect ratio;
- custom resolution;
- long side / short side sizing;
- сглаживание движения crop по видео;
- сглаживание zoom;
- offset центра crop;
- ограничение min/max zoom;
- удержание crop внутри границ исходного кадра;
- выравнивание выходного разрешения по `divisible_by`;
- дополнительную маску, которая кропается вместе с изображением;
- визуализацию области crop;
- metadata для последующего uncrop.

---

## Inputs

### `images`

Тип: `IMAGE`

Основной вход изображений.

Ожидает batch изображений в формате ComfyUI:

```text
[B, H, W, C]
```

Для видео каждый кадр обычно идет как отдельный элемент batch.

---

### `crop_mask`

Тип: `MASK`

Главная маска, по которой рассчитывается crop. Обязательна только при `tracking_mode = mask`; в `face_detection` может быть не подключена.

Именно эта маска определяет:

- где находится объект;
- где центр crop;
- какого размера должен быть crop;
- как будет двигаться crop по кадрам.

Важно: `crop_mask` влияет на сам процесс кадрирования.

---

### `masks`

Тип: `MASK`  
Опциональный вход.

Дополнительная маска, которая **не влияет на расчет crop**.

Она просто кропается тем же transform, что и изображение и `crop_mask`.

Используется, если нужно вместе с изображением передать дополнительную маску для дальнейшей генерации, композитинга или обработки.

---

## Outputs

### `cropped_images`

Тип: `IMAGE`

Результат кадрирования изображений.

Это batch cropped-кадров с разрешением, рассчитанным по настройкам ноды.

---

### `cropped_masks`

Тип: `MASK`

Кропнутая версия `crop_mask`.

То есть:

```text
crop_mask → cropped_masks
```

Эта маска соответствует cropped-изображению и может использоваться в дальнейших нодах.

---

### `masks`

Тип: `MASK`

Кропнутая версия дополнительного входа `masks`.

Если дополнительная маска не подключена, выход может дублировать `cropped_masks` в зависимости от реализации файла.

---

### `visualize`

Тип: `IMAGE`

Полноразмерные исходные изображения с нарисованной рамкой crop-области.

Используется для проверки:

- где именно находится crop;
- не выходит ли crop за границы кадра;
- насколько стабильно движется область crop;
- правильно ли работает `margin_scale`, `offset`, `fit_frame_bounds`.

---

### `crop_metadata`

Тип: `BBOXES`

Служебные данные для последующего `Uncrop`.

### `bboxes`

Тип: `BOUNDING_BOX`

Совместимый с нативными bbox-входами ComfyUI выход: для каждого кадра возвращает `[{"x", "y", "width", "height"}]`. Он описывает итоговое окно crop после smoothing, offset и `fit_frame_bounds`, поэтому его можно подключить, например, к `MaskVidExperiments Subject Uncrop`.

Для точного сшивания с этой нодой используйте всё же `crop_metadata`: affine crop допускает дробные координаты, которые формат bbox округляет до пикселей.

В metadata сохраняется информация о каждом кадре:

- исходный размер;
- crop size;
- center;
- scale;
- affine transform;
- параметры crop.

Этот выход нужно подключать в `Batch Image Uncrop By Mask Advanced`.

---

### `pipe`

Тип: `CROP_PIPE`

Служебный набор настроек и покадровых transforms. Подключайте к `pipe` другой Crop-ноды, чтобы повторить тот же crop без повторного расчёта.

---

### `report`

Тип: `STRING`

Короткий отчёт о crop, трекинге и степени увеличения.

---

### `canvas_width`

Тип: `INT`

Итоговая ширина cropped-кадра в пикселях.

---

### `canvas_height`

Тип: `INT`

Итоговая высота cropped-кадра в пикселях.

---

# Параметры Crop-ноды

## `aspect_ratio`

Задает соотношение сторон crop.

Доступные варианты обычно:

```text
1:1
16:9
9:16
4:3
3:4
4:5
5:4
2:3
3:2
21:9
9:21
```

Работает, если `use_custom_resolution = false`.

Пример:

- `16:9` — горизонтальный crop;
- `9:16` — вертикальный crop;
- `1:1` — квадратный crop;
- `4:3` — классический кадр.

Чтобы задать любое другое соотношение, включите `use_custom_aspect_ratio` и укажите `custom_aspect_ratio` в формате `W:H`, например `2.39:1`.

---

## `output_resolution_side`

Базовый размер crop.

В зависимости от `use_long_side` этот параметр управляет либо длинной, либо короткой стороной.

---

## `use_long_side`

Boolean.

Определяет, как интерпретировать `output_resolution_side`.

### Если `true`

`output_resolution_side` задает длинную сторону crop.

Пример:

```text
aspect_ratio = 16:9
output_resolution_side = 1024
```

Результат примерно:

```text
1024 × 576
```

### Если `false`

`output_resolution_side` задает короткую сторону crop.

Пример:

```text
aspect_ratio = 16:9
output_resolution_side = 576
```

Результат примерно:

```text
1024 × 576
```

Полезно, если нужно контролировать именно высоту или ширину короткой стороны.

---

## `use_custom_resolution`

Boolean.

Переключает режим задания разрешения crop.

### Если `false`

Размер crop считается автоматически через:

- `aspect_ratio`;
- `output_resolution_side`;
- `use_long_side`;
- `divisible_by`.

### Если `true`

Нода игнорирует `aspect_ratio`, `output_resolution_side` и `use_long_side`, а использует значения:

- `width`;
- `height`.

---

## `width`

Ширина crop при `use_custom_resolution = true`.

Например:

```text
width = 832
height = 480
```

Результат crop будет близок к:

```text
832 × 480
```

С учетом `divisible_by`, если он включен.

---

## `height`

Высота crop при `use_custom_resolution = true`.

Работает вместе с `width`.

---

## `divisible_by`

Заставляет обе стороны выходного crop-разрешения делиться на указанное число без остатка.

Примеры:

```text
divisible_by = 1   → без ограничений
divisible_by = 8   → ширина и высота кратны 8
divisible_by = 16  → ширина и высота кратны 16
divisible_by = 32  → ширина и высота кратны 32
```

Полезно для моделей, которые требуют размеры, кратные 8/16/32/64.

Важно: при включенном `divisible_by` соотношение сторон может быть не математически идеальным, а близким. Приоритет — чтобы обе стороны были кратны заданному значению.

---

## `margin_scale`

Множитель запаса вокруг bbox маски.

Нода сначала находит bbox объекта по `crop_mask`, затем расширяет его через `margin_scale`.

Пример:

```text
margin_scale = 1.0
```

Crop примерно соответствует bbox маски.

```text
margin_scale = 1.5
```

Crop берет область на 50% больше bbox.

```text
margin_scale = 2.0
```

Crop берет в два раза более широкую область вокруг объекта.

Используется для добавления воздуха вокруг лица, головы, тела или объекта.

---

## `smooth_center`

Boolean.

Включает сглаживание движения центра crop между кадрами.

Полезно для видео, где маска немного дрожит.

### Если `true`

Центр crop плавно интерполируется от предыдущего кадра к текущему.

### Если `false`

Центр crop сразу следует за текущей маской.

---

## `center_smoothing_strength`

Сила сглаживания центра crop.

Диапазон:

```text
0.0 – 1.0
```

Важно: в этой реализации значение работает как коэффициент следования за текущим положением.

```text
0.0
```

Центр почти остается на предыдущем положении.

```text
0.25
```

Плавное движение, сильное сглаживание.

```text
0.5
```

Среднее сглаживание.

```text
1.0
```

Центр полностью следует текущей маске, сглаживания фактически нет.

Рекомендации:

- для стабильного face crop: `0.05–0.25`;
- для быстрых движений: `0.3–0.6`;
- если crop сильно отстает: увеличить значение;
- если crop дрожит: уменьшить значение.

---

## `smooth_zoom`

Boolean.

Включает сглаживание масштаба crop между кадрами.

Полезно, если маска меняет размер от кадра к кадру и crop начинает “дышать” или пульсировать.

---

## `zoom_smoothing_strength`

Сила сглаживания zoom.

Диапазон:

```text
0.0 – 1.0
```

```text
0.0
```

Zoom почти фиксируется на предыдущем значении.

```text
0.25
```

Плавная адаптация zoom.

```text
1.0
```

Zoom сразу следует текущей маске.

Рекомендации:

- если crop пульсирует: уменьшить значение;
- если crop не успевает за реальным изменением масштаба объекта: увеличить значение;
- для face tracking часто хорошо: `0.0–0.25`.

---

## `offset_x`

Смещение центра crop по горизонтали в пикселях исходного изображения.

```text
offset_x > 0
```

Сдвигает crop вправо.

```text
offset_x < 0
```

Сдвигает crop влево.

Полезно, если нужно держать объект не строго по центру, а немного левее/правее.

---

## `offset_y`

Смещение центра crop по вертикали в пикселях исходного изображения.

```text
offset_y > 0
```

Сдвигает crop вниз.

```text
offset_y < 0
```

Сдвигает crop вверх.

Полезно, например, если bbox лица слишком низко/высоко или нужно добавить больше пространства над головой.

---

## `min_zoom`

Минимальный zoom.

Ограничивает, насколько crop может “отдалиться”.

Если объект или маска очень большие, crop может захотеть взять слишком большую область. `min_zoom` не даст scale стать слишком маленьким.

Практически:

- выше `min_zoom` → crop меньше отдаляется;
- ниже `min_zoom` → crop может брать больше пространства.

---

## `max_zoom`

Максимальный zoom.

Ограничивает, насколько crop может “приблизиться”.

Если маска стала очень маленькой, crop может резко приблизиться. `max_zoom` ограничивает такое поведение.

Практически:

- ниже `max_zoom` → меньше риск резкого zoom-in;
- выше `max_zoom` → crop может сильнее приближаться к маленькой маске.

Для face tracking этот параметр помогает бороться со схлопыванием crop, если маска временно стала маленькой.

---

## `interpolation`

Метод интерполяции при crop.

Варианты:

```text
bilinear
bicubic
```

### `bilinear`

Быстрее, обычно достаточно для видео.

### `bicubic`

Может дать более мягкое или качественное масштабирование, но потенциально медленнее.

---

## `fit_frame_bounds`

Boolean.

Если включен, crop-окно не выходит за границы исходного кадра.

### Если `false`

Crop может выходить за пределы изображения. Пустые области будут заполнены черным.

### Если `true`

Нода старается сдвинуть или скорректировать crop так, чтобы вся область была внутри исходного кадра.

Полезно, если объект находится у края кадра и не хочется получать черные поля.

Важно: при включенном `fit_frame_bounds` crop может сместиться относительно объекта, потому что нода будет приоритетно удерживать область внутри изображения.

---

# Типичные настройки Crop

## Стабильный face crop

```text
margin_scale = 1.5–2.0
smooth_center = true
center_smoothing_strength = 0.05–0.25
smooth_zoom = true
zoom_smoothing_strength = 0.0–0.2
fit_frame_bounds = true
divisible_by = 16
```

---

## Быстрое движение объекта

```text
center_smoothing_strength = 0.3–0.6
zoom_smoothing_strength = 0.2–0.5
```

Если crop отстает за движением — увеличивать `center_smoothing_strength`.

---

## Минимум пульсации zoom

```text
smooth_zoom = true
zoom_smoothing_strength = 0.0–0.15
```

---

# 2. Batch Image Uncrop By Mask Advanced

## Назначение

**Batch Image Uncrop By Mask Advanced** вставляет cropped-изображения обратно в исходное изображение/видео. Для позиционирования можно использовать точную `crop_metadata` или совместимый вход `bboxes`.

Нода нужна после обработки cropped-кадров, например:

```text
original video → crop → generation/inpaint/detailing → uncrop → full-frame result
```

Uncrop использует affine metadata, чтобы вернуть crop обратно в исходную позицию.

---

## Inputs

### `cropped_images`

Тип: `IMAGE`

Изображения, которые нужно вернуть обратно в полный кадр.

Обычно это результат обработки `cropped_images` из Crop-ноды.

---

### `crop_metadata`

Тип: `BBOXES`

Metadata из Crop-ноды.

Это предпочтительный вход: он хранит дробный affine transform и позволяет наиболее точно обратить Crop.

---

### `bboxes`

Тип: `BOUNDING_BOX`
Опциональный вход.

Принимает формат `[[{"x", "y", "width", "height"}], ...]`, который экспортирует Crop-нода. Используется, если `crop_metadata` не подключена. В этом режиме обязательно нужны `base_images`; crop масштабируется и вставляется в прямоугольник bbox. При одновременном подключении обоих входов приоритет имеет более точная `crop_metadata`.

Без `crop_metadata` или `bboxes` Uncrop не знает:

- куда возвращать crop;
- какого размера был исходный кадр;
- какой transform использовался при crop.

---

### `base_images`

Тип: `IMAGE`  
Опциональный вход.

Исходные изображения/кадры, поверх которых будет вставлен crop.

Это основной современный вход для background/base frame.

---

### `original_images`

Тип: `IMAGE`  
Опциональный вход.

Legacy alias для `base_images`.

Нужен для совместимости со старыми workflow.

Если `base_images` не подключен, но подключен `original_images`, нода использует `original_images` как базу.

---

### `crop_masks`

Тип: `MASK`  
Опциональный вход.

Маска, используемая для alpha compositing в режиме `overlay_by_mask`.

Может быть:

- `cropped_masks` из Crop-ноды;
- любая другая маска в crop-пространстве;
- маска после дополнительной обработки.

Если используется mask-based overlay, эта маска определяет, какие части cropped image будут вставлены в base image.

Перед feather нода сама расширяет эту маску через dilation на `mask_expand_px`.

---

## Output

### `images`

Тип: `IMAGE`

Финальный full-frame результат после обратной вставки crop.

---

# Параметры Uncrop-ноды

## `mode`

Режим вставки crop.

Варианты:

```text
overlay_by_mask
overlay_full
```

Значение по умолчанию: `overlay_by_mask`.

---

### `overlay_full`

Вставляет весь warped crop поверх base image.

Alpha-mask не используется.

Подходит, если весь crop был обработан и должен полностью заменить соответствующую область в оригинале.

Минус: если обработанный crop отличается по цвету/шуму/контрасту от base image, могут быть видны прямоугольные швы по границам crop.

---

### `overlay_by_mask`

Вставляет crop через alpha-mask.

Поведение зависит от `use_crop_canvas_mask`.

Если `use_crop_canvas_mask = false`, используется `crop_masks`: сначала dilation, затем feather.

Если `use_crop_canvas_mask = true`, используется отдельная прямоугольная alpha-mask всего crop с настраиваемыми inset/fade по сторонам.

---

## `blend`

Общая сила смешивания результата.

Диапазон:

```text
0.0 – 1.0
```

```text
blend = 0.0
```

Полностью оставить base image.

```text
blend = 0.5
```

Смешать base image и вставленный crop наполовину.

```text
blend = 1.0
```

Полностью применить вставку по выбранной alpha-mask.

---

## `border_blending`

Legacy-параметр сглаживания края.

Используется как запасной способ вычисления feather, если `feather_radius = 0`.

Практически лучше использовать `feather_radius`, потому что он задается прямо в пикселях.

---

## `feather_radius`

Радиус смягчения уже расширенной маски для режима, когда `use_crop_canvas_mask = false`.

Значение по умолчанию: `16`.

То есть влияет на обычный mask-based uncrop через `crop_masks`.

```text
feather_radius = 0
```

Нет дополнительного размытия, но может включиться legacy `border_blending`.

```text
feather_radius = 8
```

Мягкая граница маски.

```text
feather_radius = 16–32
```

Более широкий мягкий переход.

Важно: этот параметр не является основным fade для square-mask режима. Для square-mask используются отдельные параметры `square_mask_fade_*`.

---

## `mask_expand_px`

Радиус dilation для `crop_masks` в пикселях crop canvas. Выполняется до `feather_radius`, поэтому это не то же самое, что размытие: dilation увеличивает непрозрачную область, а feather смягчает новый край.

Значение по умолчанию: `16`.

Не применяется при `use_crop_canvas_mask = true` и в режиме `overlay_full`.

---

## `crop_rescale`

Legacy-параметр.

Используется только в старой bbox-ветке uncrop. В актуальном affine/v2 workflow обычно не влияет.

Для современных workflow с `crop_metadata` от текущей Crop-ноды этот параметр чаще всего можно оставить `1.0`.

---

## `use_crop_canvas_mask`

Boolean.

Определяет, использовать ли прямоугольную alpha-mask вместо `crop_masks`.

При переключении этого параметра интерфейс динамически изменяет высоту ноды: настройки `square_mask_*` отображаются только при `true`.

### Если `false`

Uncrop использует `crop_masks`.

Плюсы:

- можно точно вставлять только объект;
- меньше риск прямоугольных швов.

Минусы:

- если маска не совпадает с новым сгенерированным crop, результат может ломаться;
- если маска слишком узкая, могут появляться ореолы от оригинала;
- старые волосы/контуры/фон могут просвечивать на границе.

---

### Если `true`

Uncrop использует прямоугольную alpha-mask внутри crop, с отдельными настройками inset/fade для каждой стороны.

Плюсы:

- лучше подходит, если генерация изменила форму объекта;
- меньше зависит от точности старой маски;
- помогает против ореолов по краю головы/объекта.

Минусы:

- если fade/inset настроены плохо, могут появляться прямоугольные швы.

---

# Square Mask параметры

Square mask здесь фактически означает не квадратную маску, а **прямоугольную alpha-mask внутри crop-патча**.

Она строится так:

1. Берется весь crop rectangle.
2. С каждой стороны можно сделать `inset`.
3. От полученной внутренней области можно сделать плавный fade к краям.
4. Эта alpha-mask используется для вставки crop обратно в base image.

---

## `square_mask_inset_left_px`

Отступ активной области square-mask от левого края crop.

Увеличение значения сдвигает левую границу активной области вправо.

Полезно, если виден шов слева или нужно меньше затрагивать левую часть crop.

---

## `square_mask_inset_right_px`

Отступ активной области square-mask от правого края crop.

Увеличение значения сдвигает правую границу активной области влево.

Полезно, если виден шов справа.

---

## `square_mask_inset_top_px`

Отступ активной области square-mask от верхнего края crop.

Увеличение значения сдвигает верхнюю границу активной области вниз.

Полезно, если сверху виден шов или не нужно затрагивать верхнюю часть crop.

---

## `square_mask_inset_bottom_px`

Отступ активной области square-mask от нижнего края crop.

Увеличение значения сдвигает нижнюю границу активной области вверх.

Полезно, если снизу виден шов или нужно уменьшить влияние crop в нижней части.

---

## Чем inset отличается от fade

### `inset`

Определяет, **где начинается активная область маски**.

То есть inset физически сдвигает границу прямоугольной области внутрь crop.

Пример:

```text
square_mask_inset_left_px = 20
```

Первые 20 пикселей слева будут исключены из основной активной области.

---

### `fade`

Определяет, **насколько мягко край маски растворяется**.

Fade не столько двигает границу, сколько делает плавный переход от alpha 1 к alpha 0.

Пример:

```text
square_mask_fade_left_px = 20
```

Левая граница будет растворяться на протяжении примерно 20 пикселей.

---

### Простая формула

```text
inset = где находится край
fade = насколько мягкий этот край
```

Если шов находится слишком близко к краю crop — увеличивай `inset`.

Если шов слишком резкий — увеличивай `fade`.

---

## `square_mask_fade_left_px`

Ширина плавного перехода для левой стороны square-mask.

Больше значение — мягче переход слева.

---

## `square_mask_fade_right_px`

Ширина плавного перехода для правой стороны square-mask.

Больше значение — мягче переход справа.

---

## `square_mask_fade_top_px`

Ширина плавного перехода для верхней стороны square-mask.

Больше значение — мягче переход сверху.

---

## `square_mask_fade_bottom_px`

Ширина плавного перехода для нижней стороны square-mask.

Больше значение — мягче переход снизу.

---

# Типичные настройки Uncrop

## Если появляются ореолы вокруг головы/объекта

Попробовать:

```text
mode = overlay_by_mask
use_crop_canvas_mask = true
blend = 1.0
```

И настроить прямоугольную маску:

```text
square_mask_inset_left_px = 8–24
square_mask_inset_right_px = 8–24
square_mask_inset_top_px = 0–16
square_mask_inset_bottom_px = 8–32

square_mask_fade_left_px = 12–32
square_mask_fade_right_px = 12–32
square_mask_fade_top_px = 0–24
square_mask_fade_bottom_px = 8–32
```

---

## Если видны вертикальные швы по бокам crop

Увеличить:

```text
square_mask_inset_left_px
square_mask_inset_right_px
square_mask_fade_left_px
square_mask_fade_right_px
```

Например:

```text
left inset = 20
right inset = 20
left fade = 24
right fade = 24
```

---

## Если виден шов снизу

Увеличить:

```text
square_mask_inset_bottom_px
square_mask_fade_bottom_px
```

---

## Если верх кадра не нужно трогать

Поставить:

```text
square_mask_inset_top_px = 0
square_mask_fade_top_px = 0
```

или наоборот увеличить inset, если верхняя область не должна участвовать в композе.

---

## Если обычная маска хорошо совпадает с результатом

Можно использовать:

```text
mode = overlay_by_mask
use_crop_canvas_mask = false
crop_masks = cropped_masks
feather_radius = 8–24
```

Это даст более точный композит по форме маски.

---

## Если processed crop должен заменить весь прямоугольный участок

Использовать:

```text
mode = overlay_full
blend = 1.0
```

Но могут быть видны прямоугольные границы, если crop отличается по цвету/шуму от оригинала.

---

# Рекомендуемый workflow

## Базовый workflow

```text
images
  ↓
Batch Image Crop By Mask Advanced
  outputs:
    cropped_images
    cropped_masks
    masks
    visualize
    crop_metadata
  ↓
processing / generation / inpaint / detailer
  ↓
Batch Image Uncrop By Mask Advanced
  ↓
final full-frame images
```

---

## Подключения

### Crop

```text
images       → исходные кадры
crop_mask    → маска объекта для расчета crop
masks        → дополнительная маска, если нужна
```

### Uncrop

```text
cropped_images → обработанные cropped кадры
crop_metadata  → metadata из Crop
base_images    → исходные кадры
crop_masks     → cropped_masks или другая crop-space маска
```

---

# Частые проблемы и решения

## Crop прыгает по кадру

Уменьшить:

```text
center_smoothing_strength
```

Включить:

```text
smooth_center = true
```

---

## Crop отстает за объектом

Увеличить:

```text
center_smoothing_strength
```

---

## Crop пульсирует по масштабу

Уменьшить:

```text
zoom_smoothing_strength
```

или включить:

```text
smooth_zoom = true
```

---

## Crop слишком близко приближается при маленькой маске

Уменьшить:

```text
max_zoom
```

или увеличить:

```text
margin_scale
```

---

## Crop выходит за границы кадра и появляются черные поля

Включить:

```text
fit_frame_bounds = true
```

---

## Нужен размер, совместимый с моделью

Поставить:

```text
divisible_by = 8 / 16 / 32 / 64
```

---

## Uncrop дает ореол от старого объекта

Попробовать:

```text
mode = overlay_by_mask
use_crop_canvas_mask = true
```

И настроить inset/fade.

---

## Uncrop дает прямоугольные швы

Увеличить соответствующие:

```text
square_mask_inset_*_px
square_mask_fade_*_px
```

Особенно по той стороне, где виден шов.

---

## Uncrop по маске не совпадает с новым объектом

Если `use_crop_canvas_mask = false`, нода использует `crop_masks`.

Если processed crop сильно изменил форму объекта, старая маска может уже не совпадать.

Решение:

```text
use_crop_canvas_mask = true
```

или использовать более подходящую crop-space mask.

---

# Краткая шпаргалка

## Crop

```text
crop_mask              — маска, по которой считается crop
masks                  — дополнительная маска, просто кропается вместе с image
aspect_ratio           — соотношение сторон crop
output_resolution_side — базовый размер стороны
use_long_side          — true: long side, false: short side
use_custom_resolution  — использовать width/height вручную
width / height         — custom crop size
margin_scale           — запас вокруг bbox маски
smooth_center          — сглаживать движение центра
center_smoothing       — сила следования центра за текущей маской
smooth_zoom            — сглаживать zoom
zoom_smoothing         — сила следования zoom за текущей маской
offset_x / offset_y    — смещение центра crop
min_zoom               — ограничение минимального zoom
max_zoom               — ограничение максимального zoom
fit_frame_bounds       — не выходить за границы исходного кадра
divisible_by           — сделать стороны crop кратными числу
visualize              — просмотр рамки crop
crop_metadata          — данные для uncrop
```

## Uncrop

```text
cropped_images                 — обработанный crop
crop_metadata                  — metadata из Crop
base_images / original_images  — исходный full-frame фон
crop_masks                     — маска для overlay_by_mask
mode                           — full overlay или overlay по маске
blend                          — сила смешивания
feather_radius                 — размытие обычной crop mask
use_crop_canvas_mask           — использовать прямоугольную alpha mask всего crop
square_mask_inset_*            — отступ активной области от каждой стороны
square_mask_fade_*             — мягкость края каждой стороны
```
