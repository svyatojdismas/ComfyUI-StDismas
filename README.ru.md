# ComfyUI-StDismas — русская документация

Пакет нод для ComfyUI с покадровым кадрированием изображений и видео, mask-based compositing, optional face tracking и вспомогательными VHS-инструментами.

<p align="center"><a href="README.md">English</a> | <a href="README.ru.md">Russian</a></p>

## Документация

- [Подробная документация Batch Image Crop / Uncrop By Mask Advanced](nodes/README_batch_crop_uncrop_by_mask_advanced_ru.md)

## Кратко о Crop / Uncrop

`Batch Image Crop By Mask Advanced (StDismas)` умеет работать по обычной маске или по face detector, сглаживать траекторию, удерживать identity выбранного лица и экспортировать точные `crop_metadata` либо совместимые `bboxes` типа `BOUNDING_BOX`.

`Batch Image Uncrop By Mask Advanced (StDismas)` возвращает обработанный crop в исходный кадр. Доступны точный affine-путь через `crop_metadata`, bbox-путь через `bboxes`, mask-based stitching с dilation/feather, прямоугольная маска всего crop canvas и режим вставки всего crop.

Для подробных параметров, рекомендуемых настроек и решений типовых проблем используйте [полное русское руководство](nodes/README_batch_crop_uncrop_by_mask_advanced_ru.md).
