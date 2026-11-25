import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


BACKGROUND = "#0d141d"
SURFACE = "#151f2b"
TRACK = "#213043"
BORDER = "#334558"
INK = "#eff6ff"
MUTED = "#a6b8cc"
ACCENT = "#77e1d4"
AMBER = "#ffb76b"
TILE_COLORS = (
    (56, 173, 202), (119, 108, 235), (231, 151, 83), (57, 175, 140),
    (220, 111, 161), (82, 144, 237), (177, 174, 88), (157, 116, 199),
    (232, 109, 95), (86, 188, 181), (198, 144, 108), (134, 173, 100),
)


def fade_axis(tile, fade):
    mask = np.ones(tile.size, dtype=np.float32)
    fade = min(tile.overlap, fade)
    frozen = tile.overlap - fade
    mask[:frozen] = 0
    mask[frozen:tile.overlap] = np.linspace(0, 1, fade, dtype=np.float32)
    return mask


def stitch_weights(size, curve, mode):
    weights = np.linspace(0, 1, size, dtype=np.float32)
    if curve == "overwrite":
        return np.full_like(weights, float(mode == "later"))
    if curve == "midpoint":
        weights = (weights >= 0.5).astype(np.float32)
    elif curve == "smoothstep":
        weights = weights * weights * (3 - 2 * weights)
    return 1 - weights if mode == "later" else weights


def spatial_maps(layout, pixels_per_cell):
    sp = layout.spatial or {}
    fade_x = int(sp.get("fade_width", 0)) // pixels_per_cell
    fade_y = int(sp.get("fade_height", 0)) // pixels_per_cell
    mode, curve = sp.get("overlap_mode", "earlier"), sp.get("overlap_blend", "linear")
    cols = [(col, fade_axis(col, fade_x), stitch_weights(col.overlap, curve, mode)) for col in layout.columns]
    rows = [(row, fade_axis(row, fade_y), stitch_weights(row.overlap, curve, mode)) for row in layout.rows]
    sampling = np.ones((layout.height, layout.width), dtype=np.float32)
    stitched = np.zeros((layout.height, layout.width, 3), dtype=np.float32)

    for i, (row, mask_y, weights_y) in enumerate(rows):
        for j, (col, mask_x, weights_x) in enumerate(cols):
            y = slice(row.start, row.start + row.size)
            x = slice(col.start, col.start + col.size)
            mask = sampling[y, x]
            np.minimum(mask, mask_x[None, :], out=mask)
            np.minimum(mask, mask_y[:, None], out=mask)

            color = np.asarray(TILE_COLORS[(i * len(cols) + j) % len(TILE_COLORS)], dtype=np.float32)
            region = stitched[y, x]
            # MMH3 blends its left strip first, then its top strip, including the shared corner.
            left = region[:, :col.overlap]
            top = region[:row.overlap, :]
            left += (color - left) * weights_x[None, :, None]
            top += (color - top) * weights_y[:, None, None]
            region[row.overlap:, col.overlap:] = color

    return sampling, stitched


def font(size, bold=False):
    fonts = ("segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "DejaVuSans.ttf")
    for name in fonts:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def sampling_image(mask, noise_floor):
    # Brighter orange depicts lower mask values; green marks the transition to free sampling.
    free = np.array((26, 48, 68))
    fade = np.array((80, 182, 169))
    locked = free + (np.array((248, 166, 82)) - free) * (1 - noise_floor)
    fade = free + (fade - free) * (1 - noise_floor)
    mask = mask[..., None]
    shade = np.where(mask <= 0.5,
                     locked + (fade - locked) * (2 * mask),
                     fade + (free - fade) * (2 * mask - 1))
    return Image.fromarray(shade.astype(np.uint8))


def resize_frame(frame, size):
    rgb = frame[..., :3].detach().float().cpu().numpy()
    return Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS)


def video_overlay(layout, size, pixels_per_cell, show_overlap_and_fade):
    overlay = Image.new("RGBA", size)
    if show_overlap_and_fade:
        mask, _ = spatial_maps(layout, pixels_per_cell)
        noise_floor = float((layout.spatial or {}).get("masked_area_noise", 0))
        if noise_floor < 1:
            tint = sampling_image(mask, noise_floor)
            opacity = np.where(mask < 1, round(114 * (1 - noise_floor)), 0).astype(np.uint8)
            tint.putalpha(Image.fromarray(opacity))
            overlay = tint.resize(size, Image.Resampling.NEAREST)
    edges = Image.new("RGBA", size)
    paint_boundaries(ImageDraw.Draw(edges), layout, 0, 0, *size, font(16, True), pixels_per_cell)
    shadow = Image.new("RGBA", size)
    shadow.putalpha(edges.getchannel("A").filter(ImageFilter.MaxFilter(5)).point(lambda value: value * 0.7))
    overlay.alpha_composite(shadow)
    overlay.alpha_composite(edges)
    return overlay


def measured_overlaps(tiles, pixels_per_cell):
    values = [tile.overlap * pixels_per_cell for tile in tiles[1:]]
    if not values:
        return "0"
    if min(values) == max(values):
        return str(values[0])
    if len(values) <= 4:
        return ", ".join(map(str, values))
    return f"{min(values)}–{max(values)}"


def paint_boundaries(draw, layout, x, y, width, height, label_font, pixels_per_cell):
    compact_font = font(12, True)

    def sx(position):
        return x + round(position * width / layout.width)

    def sy(position):
        return y + round(position * height / layout.height)

    draw.rectangle((x, y, x + width - 1, y + height - 1), outline=INK, width=2)
    for col in layout.columns[1:]:
        draw.line((sx(col.start), y, sx(col.start), y + height - 1), fill=INK, width=2)
    for row in layout.rows[1:]:
        draw.line((x, sy(row.start), x + width - 1, sy(row.start)), fill=INK, width=2)

    # Interior end edges complete the sampled rectangles without obscuring their solid start edges.
    for col in layout.columns[:-1]:
        edge = sx(col.start + col.size)
        for top in range(y, y + height, 10):
            draw.line((edge, top, edge, min(top + 4, y + height - 1)), fill=INK)
    for row in layout.rows[:-1]:
        edge = sy(row.start + row.size)
        for left in range(x, x + width, 10):
            draw.line((left, edge, min(left + 4, x + width - 1), edge), fill=INK)

    for i, row in enumerate(layout.rows):
        row_end = layout.rows[i + 1].start if i + 1 < len(layout.rows) else layout.height
        for j, col in enumerate(layout.columns):
            col_end = layout.columns[j + 1].start if j + 1 < len(layout.columns) else layout.width
            area_w, area_h = sx(col_end) - sx(col.start), sy(row_end) - sy(row.start)
            tile = i * len(layout.columns) + j + 1
            number = f"{tile:02d}"
            size = f"{col.size * pixels_per_cell}×{row.size * pixels_per_cell}"
            label = f"{number}  {size}"
            if draw.textlength(label, font=label_font) + 22 <= area_w:
                lines = ((label, label_font),)
            elif draw.textlength(size, font=compact_font) + 22 <= area_w and area_h >= 48:
                lines = ((number, label_font), (size, compact_font))
            else:
                lines = ((number, label_font),)
            label_w = round(max(draw.textlength(line, font=typeface) for line, typeface in lines)) + 12
            label_h = 7 + 16 * len(lines)
            if label_w + 10 <= area_w and label_h + 10 <= area_h:
                left, top = sx(col.start) + 6, sy(row.start) + 6
                draw.rounded_rectangle((left, top, left + label_w, top + label_h), 5, fill=BACKGROUND)
                for line_index, (line, typeface) in enumerate(lines):
                    draw.text((left + 6, top + 11 + 16 * line_index), line, font=typeface, fill=INK, anchor="lm")


def draw_timeline(draw, bounds, layout, body_font, label_font, heading_font, show_overlap):
    x, y, right, bottom = bounds
    draw.rounded_rectangle(bounds, 12, fill=SURFACE, outline=BORDER)
    draw.text((x + 20, y + 16), "Temporal chunks", font=heading_font, fill=INK)
    layout_frames = layout.frames
    if layout_frames is None:
        draw.text((x + 20, y + 49), "Connect the input latent or set planned_frames to show chunk boundaries.",
                  font=body_font, fill=MUTED)
        return

    subtitle = f"{layout_frames} frames / {layout.duration_source}"
    if layout.temporal is not None:
        tp = layout.temporal
        subtitle += f"  ·  {tp['chunk_length']} frame chunks"
        if show_overlap:
            subtitle += f"  /  {tp['temporal_overlap']} overlap"
        subtitle += f"  /  anchor {tp['anchor_strength']:g}"
    draw.text((x + 20, y + 45), subtitle, font=body_font, fill=MUTED)
    if show_overlap and len(layout.chunks) > 1:
        draw.text((right - 20, y + 20), "Amber: shared frames", font=body_font, fill=AMBER, anchor="rt")

    left, end = x + 63, right - 176
    top = y + 74
    previous_end = 0
    for i, chunk in enumerate(layout.chunks):
        overlap_end = min(previous_end, chunk.frame_end)
        previous_end = max(previous_end, chunk.frame_end)
        if len(layout.chunks) > 10 and 4 <= i < len(layout.chunks) - 4:
            if i == 4:
                draw.text((left, top + 7), f"{len(layout.chunks) - 8} additional chunks in layout_report",
                          font=label_font, fill=MUTED, anchor="lm")
                top += 24
            continue

        x0 = left + round((end - left) * chunk.frame_start / layout_frames)
        x1 = max(x0 + 1, left + round((end - left) * chunk.frame_end / layout_frames))
        draw.rounded_rectangle((left, top, end, top + 14), 4, fill=TRACK)
        draw.rounded_rectangle((x0, top, x1, top + 14), 3, fill=TILE_COLORS[i % len(TILE_COLORS)])
        if show_overlap and overlap_end > chunk.frame_start:
            shared = left + round((end - left) * overlap_end / layout_frames)
            draw.rounded_rectangle((x0, top, max(x0 + 1, shared), top + 14), 3, fill=AMBER)
        draw.text((x + 20, top + 7), f"{i + 1:02d}", font=label_font, fill=INK, anchor="lm")
        draw.text((end + 14, top + 7), f"[{chunk.frame_start}, {chunk.frame_end})  ·  {chunk.frame_end - chunk.frame_start}f",
                  font=label_font, fill=INK, anchor="lm")
        top += 24

    draw.text((left, bottom - 17), "0", font=label_font, fill=MUTED, anchor="lm")
    draw.text((end, bottom - 17), f"{layout_frames} frames", font=label_font, fill=MUTED, anchor="rm")


def overlap_panels(layout, pixels_per_cell):
    sp = layout.spatial or {}
    sampling, stitched = spatial_maps(layout, pixels_per_cell)
    noise_floor = float(sp.get("masked_area_noise", 0))
    fade_x, fade_y = sp.get("fade_width", 0), sp.get("fade_height", 0)
    mask_detail = f"Fade X/Y {fade_x}/{fade_y} px  ·  {noise_floor:.0%} minimum mask"
    schedule = sp.get("dynamic_fade", "off")
    mask_note = "Orange: protected   ·   Green: fade   ·   Blue: free"
    if schedule != "off":
        minimum = int(sp.get("dynamic_fade_min", 0))
        low_x, low_y = min(minimum, fade_x), min(minimum, fade_y)
        if (low_x, low_y) != (fade_x, fade_y):
            high, low = f"{fade_x}/{fade_y}", f"{low_x}/{low_y}"
            first, last = (low, high) if schedule == "widening" else (high, low)
            mask_detail = f"Fade X/Y {first} → {last} px  ·  {noise_floor:.0%} minimum mask"
            mask_note = "Orange: protected   ·   Green: max fade   ·   Blue: free"
    stitch_detail = f"{sp['overlap_mode']} / {sp['overlap_blend']}  ·  blends left, then top" if sp else "Whole-frame sampling"
    if sp.get("brightness_match", False):
        stitch_detail += "  ·  brightness match"

    panels = (
        ("Tile bounds & noise mask", mask_detail, sampling_image(sampling, noise_floor)),
        ("Stitch weights", stitch_detail, Image.fromarray(stitched.astype(np.uint8))),
    )
    return panels, mask_note


def render_preview(layout, background, pixels_per_cell, show_overlap_and_fade=True):
    margin, gap = 22, 16
    scale = min(620 / layout.width, 405 / layout.height)
    map_w, map_h = max(1, round(layout.width * scale)), max(1, round(layout.height * scale))
    if show_overlap_and_fade:
        panels, mask_note = overlap_panels(layout, pixels_per_cell)
        panel_w = max(450, map_w + 36)
        map_y, footer_h = 162, 84
    else:
        panels = (("Tile bounds", "", None),)
        panel_w = max(700, map_w + 36)
        map_y, footer_h = 148, 43
    canvas_w = 2 * margin + len(panels) * panel_w + (len(panels) - 1) * gap
    card_y = 102
    card_end = map_y + map_h + footer_h
    timeline_y = card_end + gap
    display_chunks = len(layout.chunks) if len(layout.chunks) <= 10 else 9
    timeline_h = 100 + 24 * display_chunks if display_chunks else 85
    diagram = Image.new("RGB", (canvas_w, timeline_y + timeline_h + margin), BACKGROUND)
    draw = ImageDraw.Draw(diagram)
    title_font, heading_font, body_font, label_font = font(34, True), font(20, True), font(15), font(14, True)
    draw.text((margin, 16), "MMH3 Tile Preview", font=title_font, fill=INK)
    columns, rows = len(layout.columns), len(layout.rows)
    tile_count = columns * rows
    subtitle = (f"{layout.width * pixels_per_cell} × {layout.height * pixels_per_cell} px  ·  "
                f"{columns} columns × {rows} rows  ·  {tile_count} tiles per chunk")
    if layout.chunks:
        subtitle += f"  ·  {tile_count * len(layout.chunks)} sampler calls"
    draw.text((margin + 1, 65), subtitle, font=body_font, fill=ACCENT)

    backdrop = None
    if background is not None and len(background):
        backdrop = resize_frame(background[0], (map_w, map_h))

    for i, (heading, detail, panel) in enumerate(panels):
        x = margin + i * (panel_w + gap)
        draw.rounded_rectangle((x, card_y, x + panel_w, card_end), 12, fill=SURFACE, outline=BORDER)
        draw.text((x + 18, card_y + 12), heading, font=heading_font, fill=INK)
        map_x = x + (panel_w - map_w) // 2
        if panel is None:
            panel = backdrop if backdrop is not None else Image.new("RGB", (map_w, map_h), TRACK)
        else:
            draw.text((x + 18, card_y + 38), detail, font=body_font, fill=MUTED)
            panel = panel.resize((map_w, map_h), Image.Resampling.NEAREST)
            if backdrop is not None:
                panel = Image.blend(backdrop, panel, 0.60 if i == 0 else 0.55)
        diagram.paste(panel, (map_x, map_y))
        legend_y = map_y + map_h + 18
        if i == 0:
            paint_boundaries(draw, layout, map_x, map_y, map_w, map_h, label_font, pixels_per_cell)
            draw.text((x + 18, legend_y), "White edges: solid start / dashed end", font=body_font, fill=MUTED)
            if show_overlap_and_fade:
                draw.text((x + 18, legend_y + 21), mask_note, font=body_font, fill=MUTED)
                overlap_x = measured_overlaps(layout.columns, pixels_per_cell)
                overlap_y = measured_overlaps(layout.rows, pixels_per_cell)
                draw.text((x + 18, legend_y + 42), f"Actual overlap X {overlap_x}  /  Y {overlap_y} px",
                          font=body_font, fill=ACCENT)
        else:
            draw.rectangle((map_x, map_y, map_x + map_w - 1, map_y + map_h - 1), outline=BORDER)
            swatches = min(tile_count, len(TILE_COLORS))
            per_line = (panel_w - 36) // 43
            for j in range(swatches):
                line, position = divmod(j, per_line)
                swatch_x, swatch_y = x + 18 + position * 43, legend_y + line * 21
                draw.rounded_rectangle((swatch_x, swatch_y, swatch_x + 15, swatch_y + 15), 3, fill=TILE_COLORS[j])
                draw.text((swatch_x + 19, swatch_y + 7), f"{j + 1:02d}", font=label_font, fill=INK, anchor="lm")
            note = "Colors mix in stitch regions"
            if tile_count > len(TILE_COLORS):
                note = f"Tile colors repeat every {len(TILE_COLORS)} tiles"
            draw.text((x + 18, legend_y + 21 * ((swatches - 1) // per_line + 1)), note, font=body_font, fill=MUTED)

    draw_timeline(draw, (margin, timeline_y, canvas_w - margin, timeline_y + timeline_h), layout,
                  body_font, label_font, heading_font, show_overlap_and_fade)
    return diagram
