import { app } from "../../scripts/app.js";

const CROP_NODE_CLASSES = new Set(["BatchImageCropByMaskAdvanced_StDismas"]);
const UNCROP_NODE_CLASSES = new Set(["BatchImageUncropByMaskAdvanced_StDismas"]);
const LEGACY_FACE_CROP_CLASS = "BatchImageCropByMaskOrFaceAdvanced_StDismas";
const UNIVERSAL_CROP_CLASS = "BatchImageCropByMaskAdvanced_StDismas";

const FACE_WIDGETS = [
  "face_detector",
  "face_confidence",
  "face_select",
  "identity_track",
  "identity_threshold",
  "identity_pack",
  "face_model_device",
  "keep_face_models_loaded",
  "fallback_detector",
  "fallback_head_frac",
];

const SQUARE_MASK_WIDGETS = [
  "square_mask_inset_left_px",
  "square_mask_inset_right_px",
  "square_mask_inset_top_px",
  "square_mask_inset_bottom_px",
  "square_mask_fade_left_px",
  "square_mask_fade_right_px",
  "square_mask_fade_top_px",
  "square_mask_fade_bottom_px",
  "square_mask_units",
];

// Widget order used by the first FaceRefine-upgrade build. Old ComfyUI workflows
// store values as an array, so keep this map to migrate them after UI reordering.
const LEGACY_WIDGET_ORDER = [
  "aspect_ratio",
  "output_long_side",
  "use_long_side",
  "use_custom_resolution",
  "width",
  "height",
  "margin_scale",
  "smooth_center",
  "center_smoothing_strength",
  "smooth_zoom",
  "zoom_smoothing_strength",
  "offset_x",
  "offset_y",
  "min_zoom",
  "max_zoom",
  "interpolation",
  "fit_frame_bounds",
  "divisible_by",
  "enable_visualize",
  "crop_chunk_size",
  "tracking_mode",
  "smoothing_method",
  "center_smooth_window",
  "size_smooth_window",
  "size_metric",
  "resolution_mode",
  "auto_resolution_cap",
  "face_detector",
  "face_confidence",
  "face_select",
  "identity_track",
  "identity_threshold",
  "identity_pack",
  "face_model_device",
  "keep_face_models_loaded",
  "fallback_detector",
  "fallback_head_frac",
];

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;

  if (!widget.__stdismasVisibility) {
    widget.__stdismasVisibility = {
      type: widget.type,
      computeSize: widget.computeSize,
    };
  }

  const original = widget.__stdismasVisibility;
  widget.hidden = !visible;
  widget.type = visible ? original.type : "hidden";
  widget.computeSize = visible
    ? original.computeSize
    : () => [0, -4];
}

function resizeNode(node) {
  requestAnimationFrame(() => {
    const computed = node.computeSize?.();
    if (!computed) return;
    node.setSize?.([Math.max(node.size?.[0] ?? 0, computed[0]), computed[1]]);
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
  });
}

function hookWidget(node, name, refresh) {
  const widget = findWidget(node, name);
  if (!widget || widget.__stdismasDynamicHook) return;
  widget.__stdismasDynamicHook = true;

  const originalCallback = widget.callback;
  widget.callback = function (...args) {
    const result = originalCallback?.apply(this, args);
    setTimeout(refresh, 0);
    return result;
  };

  const originalMouse = widget.mouse;
  widget.mouse = function (...args) {
    const result = originalMouse?.apply(this, args);
    setTimeout(refresh, 0);
    return result;
  };
}

function refreshCropWidgets(node) {
  const trackingMode = findWidget(node, "tracking_mode")?.value ?? "mask";
  const faceMode = trackingMode === "face_detection";
  const identityEnabled = Boolean(findWidget(node, "identity_track")?.value);
  const fallbackEnabled = (findWidget(node, "fallback_detector")?.value ?? "none") !== "none";

  for (const name of FACE_WIDGETS) {
    let visible = faceMode;
    if (name === "identity_threshold" || name === "identity_pack") {
      visible = faceMode && identityEnabled;
    } else if (name === "fallback_head_frac") {
      visible = faceMode && fallbackEnabled;
    }
    setWidgetVisible(findWidget(node, name), visible);
  }

  const method = findWidget(node, "smoothing_method")?.value ?? "gaussian";
  const smoothingEnabled = method !== "none";
  const windowedMethod = ["gaussian", "savgol", "moving_average"].includes(method);
  const centerEnabled = Boolean(findWidget(node, "smooth_center")?.value);
  const zoomEnabled = Boolean(findWidget(node, "smooth_zoom")?.value);

  setWidgetVisible(
    findWidget(node, "center_smoothing_strength"),
    smoothingEnabled && centerEnabled,
  );
  setWidgetVisible(
    findWidget(node, "center_smooth_window"),
    smoothingEnabled && windowedMethod && centerEnabled,
  );
  setWidgetVisible(
    findWidget(node, "zoom_smoothing_strength"),
    smoothingEnabled && zoomEnabled,
  );
  setWidgetVisible(
    findWidget(node, "size_smooth_window"),
    smoothingEnabled && windowedMethod && zoomEnabled,
  );

  resizeNode(node);
}

function setupCropNode(node) {
  if (node.__stdismasDynamicCrop) return;
  node.__stdismasDynamicCrop = true;

  const refresh = () => refreshCropWidgets(node);
  for (const name of [
    "tracking_mode",
    "smoothing_method",
    "smooth_center",
    "smooth_zoom",
    "identity_track",
    "fallback_detector",
  ]) {
    hookWidget(node, name, refresh);
  }
  refresh();
}

function refreshUncropWidgets(node) {
  const squareMaskEnabled = Boolean(findWidget(node, "use_square_mask")?.value);
  for (const name of SQUARE_MASK_WIDGETS) {
    setWidgetVisible(findWidget(node, name), squareMaskEnabled);
  }
  resizeNode(node);
}

function setupUncropNode(node) {
  if (node.__stdismasDynamicUncrop) return;
  node.__stdismasDynamicUncrop = true;

  const refresh = () => refreshUncropWidgets(node);
  hookWidget(node, "use_square_mask", refresh);
  refresh();
}

function installNamedWidgetState(nodeType) {
  const prototype = nodeType.prototype;
  if (prototype.__stdismasNamedWidgetState) return;
  prototype.__stdismasNamedWidgetState = true;

  const originalConfigure = prototype.onConfigure;
  prototype.onConfigure = function (info) {
    const result = originalConfigure?.apply(this, arguments);
    const saved = info?.widgets_values;
    if (!saved || !this.widgets) return result;

    const valuesByName = Array.isArray(saved)
      ? Object.fromEntries(
          LEGACY_WIDGET_ORDER.slice(0, saved.length).map((name, index) => [name, saved[index]]),
        )
      : saved;

    for (const widget of this.widgets) {
      if (Object.prototype.hasOwnProperty.call(valuesByName, widget.name)) {
        widget.value = valuesByName[widget.name];
        widget.callback?.(widget.value);
      }
    }
    return result;
  };

  const originalSerialize = prototype.onSerialize;
  prototype.onSerialize = function (info) {
    const result = originalSerialize?.apply(this, arguments);
    info.widgets_values = Object.fromEntries(
      (this.widgets ?? [])
        .filter((widget) => widget.type !== "button")
        .map((widget) => [widget.name, widget.value]),
    );
    return result;
  };
}

app.registerExtension({
  name: "stdismas.batch_crop_dynamic_widgets",
  async beforeConfigureGraph(graphData) {
    for (const node of graphData?.nodes ?? []) {
      if (node.type === LEGACY_FACE_CROP_CLASS) {
        node.type = UNIVERSAL_CROP_CLASS;
      }
    }
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (CROP_NODE_CLASSES.has(nodeData?.name)) {
      installNamedWidgetState(nodeType);
    }
  },
  nodeCreated(node) {
    if (CROP_NODE_CLASSES.has(node.comfyClass)) {
      // Let ComfyUI restore saved widget values before applying visibility.
      setTimeout(() => setupCropNode(node), 50);
    } else if (UNCROP_NODE_CLASSES.has(node.comfyClass)) {
      // Let ComfyUI restore saved widget values before applying visibility.
      setTimeout(() => setupUncropNode(node), 50);
    }
  },
});
