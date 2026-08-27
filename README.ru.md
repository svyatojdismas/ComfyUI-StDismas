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

## Installation
Clone into ComfyUI/custom_nodes/ and restart ComfyUI:
```
git clone https://github.com/svyatojdismas/ComfyUI-StDismas.git
```
