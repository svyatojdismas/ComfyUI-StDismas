# ComfyUI-StDismas — русская документация

Пакет нод для ComfyUI с покадровым кадрированием изображений и видео, mask-based compositing, optional face tracking и вспомогательными VHS-инструментами.

<p align="center"><a href="README.md">English</a> | <a href="README.ru.md">Russian</a></p>

## Документация

- [Подробная документация Batch Image Crop / Uncrop By Mask Advanced](nodes/README_batch_crop_uncrop_by_mask_advanced_ru.md)

## Кратко о Crop / Uncrop

`Batch Image Crop By Mask Advanced (StDismas)` умеет работать по обычной маске или по face detector, сглаживать траекторию, удерживать identity выбранного лица и экспортировать точные `crop_metadata` либо совместимые `bboxes` типа `BOUNDING_BOX`.

`Batch Image Uncrop By Mask Advanced (StDismas)` возвращает обработанный crop в исходный кадр. Доступны точный affine-путь через `crop_metadata`, bbox-путь через `bboxes`, mask-based stitching с dilation/feather, прямоугольная маска всего crop canvas и режим вставки всего crop.

Для подробных параметров, рекомендуемых настроек и решений типовых проблем используйте [полное русское руководство](nodes/README_batch_crop_uncrop_by_mask_advanced_ru.md).

## Необязательная интеграция с VideoHelperSuite

Если установлен [ComfyUI-VideoHelperSuite](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite), StDismas добавляет:

- `Load Video FFmpeg (Upload) Frames`: пересчёт FPS через FFmpeg с точным применением `skip_first_frames`, `select_every_nth` и `frame_load_cap`;
- пресеты полного диапазона BT.709 `video/h264-mp4-pc` и `video/nvenc_h264-mp4-pc` для стандартной VHS-ноды `Video Combine`.

Интеграция необязательна. Если VideoHelperSuite отсутствует или несовместим, остальные ноды StDismas продолжают загружаться обычным образом.

## Превью тайлов MMH3

`MMH3 Tile Preview (StDismas)` строит схему тайлов, перекрытий и зон fade, показывает веса сшивки и таймлайн чанков `MMH3 Ultimate Upscale`. Прямо в ноде можно посмотреть схему или воспроизвести видео. На выходах доступны схема `IMAGE` и текстовый `layout_report` с координатами, размерами и диапазонами кадров.

1. Разведи `spatial_split_param`, `temporal_split_param` и `latent_upscale_param` из одних и тех же [нод MMH3](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale) в апскейлер и в превью. Нода читает уже рассчитанные настройки, включая режим равных тайлов в новых версиях.
2. Подай тот же `latent`, что поступает в `MMH3 Ultimate Upscale`. Если латент уже масштабирован в другой ноде, подай результат масштабирования в обе ноды и оставь `latent_upscale_param` свободным. Превью возьмёт размер и длину клипа из латента. Чтобы спланировать раскладку с заданным разрешением заранее, заполни `planned_frames` без подключения латента. Длина округляется вверх к сетке генератора MiniMax H3 `17k+5`. При `0` доступна пространственная схема, если задано разрешение апскейла.
3. В `background_image` можно подать кадр клипа. Если на входе батч, нода берёт первый кадр и масштабирует его под пропорции целевого разрешения.

Белые линии очерчивают тайлы, номера показывают порядок сэмплирования. Оранжевым отмечены защищённые полосы перекрытий, бирюзовым — fade. Цвета на карте сшивки смешиваются по выбранному режиму, включая углы пересечения швов. Для динамического fade показана максимальная ширина полосы. Схема строится на CPU без декодирования VAE и прогона апскейла.

Выключи `show_overlap_and_fade`, чтобы увидеть чистую схему с границами и размерами тайлов и диапазонами чанков. Переключатель скрывает цветные полосы, карту сшивки и подсветку общих кадров. Численные значения перекрытий и fade доступны в `layout_report`.

Для воспроизведения выбери `preview_mode = video` и подай батч кадров в `background_image`. Подойдёт `IMAGE` из загрузчика VHS или связки `Load Video` → `Get Video Components`. Укажи частоту батча в `preview_fps`. Сетка тайлов накладывается на движущийся клип, а цвет полос регулируется `show_overlap_and_fade`. Для воспроизведения в ноде используется MP4 без звука с длинной стороной до 1280 px. Выход `preview_video` отдаёт батч кадров `IMAGE` с наложенной сеткой. Для экспорта подключи его к `VHS Video Combine` и задай там FPS.

## Установка
Клонировать репозиторий в ComfyUI/custom_nodes/ и перезапустить ComfyUI:
```
git clone https://github.com/svyatojdismas/ComfyUI-StDismas.git
```
