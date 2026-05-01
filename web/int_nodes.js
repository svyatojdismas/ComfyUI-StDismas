import { app } from "../../scripts/app.js";

function clamp(v, min, max) {
  if (min != null && v < min) v = min;
  if (max != null && v > max) v = max;
  return v;
}

function snapToRule(value, options = {}) {
  let v = Number(value);
  if (!Number.isFinite(v)) v = Number(options.default ?? 0);

  const min = options.min;
  const max = options.max;
  const step = Math.max(1, Number(options.step ?? 1));
  const mod = Number(options.mod ?? 0);

  v = clamp(v, min, max);
  v = Math.round((v - mod) / step) * step + mod;
  v = clamp(v, min, max);
  return Math.trunc(v);
}

function updateWidgetValue(node, widget, newValue) {
  const snapped = snapToRule(newValue, widget.options || {});
  widget.value = snapped;
  if (widget.options?.property && node.properties && widget.options.property in node.properties) {
    node.setProperty(widget.options.property, snapped);
  }
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas(true, true);
}

function patchValueWidget(node, valueWidget) {
  if (!valueWidget || valueWidget.__stdismasPatched) return;
  valueWidget.__stdismasPatched = true;

  const originalCallback = valueWidget.callback?.bind(valueWidget);
  valueWidget.callback = function (v, ...args) {
    const snapped = snapToRule(v, this.options || {});
    this.value = snapped;
    if (originalCallback) {
      return originalCallback(snapped, ...args);
    }
    node.graph?.setDirtyCanvas(true, true);
  };

  const originalMouse = valueWidget.mouse?.bind(valueWidget);
  valueWidget.mouse = function (event, pos, n) {
    const result = originalMouse ? originalMouse(event, pos, n) : undefined;
    const snapped = snapToRule(this.value, this.options || {});
    if (this.value !== snapped) {
      this.value = snapped;
      node.graph?.setDirtyCanvas(true, true);
    }
    return result;
  };
}

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function dimensionRule(format) {
  switch (format) {
    case "LTXV":
      return { min: 0, max: 30720, step: 32, mod: 0 };
    case "WAN":
      return { min: 0, max: 30720, step: 8, mod: 0 };
    case "F.Klein":
      return { min: 0, max: 30720, step: 16, mod: 0 };
    default:
      return { min: 0, max: 30720, step: 1, mod: 0 };
  }
}

function durationRule(format) {
  switch (format) {
    case "LTXV":
      return { min: 1, max: 2147483647, step: 8, mod: 1 };
    case "WAN":
      return { min: 1, max: 2147483647, step: 4, mod: 1 };
    default:
      return { min: 0, max: 2147483647, step: 1, mod: 0 };
  }
}

function divisibleRule(divisibleBy) {
  const d = Math.max(1, Math.trunc(Number(divisibleBy) || 1));
  return { min: 0, max: 2147483647, step: d, mod: 0 };
}

function applyRule(node, valueWidget, rule) {
  if (!valueWidget) return;
  valueWidget.options = valueWidget.options || {};
  valueWidget.options.min = rule.min;
  valueWidget.options.max = rule.max;
  valueWidget.options.step = rule.step;
  valueWidget.options.mod = rule.mod;
  updateWidgetValue(node, valueWidget, valueWidget.value);
}

function hookSourceWidget(node, sourceWidget, apply) {
  if (!sourceWidget || sourceWidget.__stdismasRuleHooked) return;
  sourceWidget.__stdismasRuleHooked = true;

  const originalCallback = sourceWidget.callback?.bind(sourceWidget);
  sourceWidget.callback = function (v, ...args) {
    const result = originalCallback ? originalCallback(v, ...args) : undefined;
    setTimeout(() => apply(), 0);
    return result;
  };

  const originalMouse = sourceWidget.mouse?.bind(sourceWidget);
  sourceWidget.mouse = function (event, pos, n) {
    const result = originalMouse ? originalMouse(event, pos, n) : undefined;
    setTimeout(() => apply(), 0);
    return result;
  };
}

function setupReactiveRule(node, kind) {
  const valueWidget = findWidget(node, "value");
  patchValueWidget(node, valueWidget);
  if (!valueWidget) return;

  let sourceWidget;
  let getRule;

  if (kind === "dimension") {
    sourceWidget = findWidget(node, "format");
    getRule = () => dimensionRule(sourceWidget?.value);
  } else if (kind === "duration") {
    sourceWidget = findWidget(node, "format");
    getRule = () => durationRule(sourceWidget?.value);
  } else if (kind === "divisible") {
    sourceWidget = findWidget(node, "divisible_by");
    patchValueWidget(node, sourceWidget);
    getRule = () => divisibleRule(sourceWidget?.value);
  }

  if (!sourceWidget || !getRule) return;

  const apply = () => applyRule(node, valueWidget, getRule());
  hookSourceWidget(node, sourceWidget, apply);
  apply();
}

app.registerExtension({
  name: "stdismas.int_nodes_builtin",
  nodeCreated(node) {
    if (node.comfyClass === "SetDimension_StDismas") {
      setupReactiveRule(node, "dimension");
    } else if (node.comfyClass === "SetDuration_StDismas") {
      setupReactiveRule(node, "duration");
    } else if (node.comfyClass === "IntDivisibleBy_StDismas") {
      setupReactiveRule(node, "divisible");
    }
  },
});
