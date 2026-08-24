import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


const NODE_NAME = "StDismas_LoadVideoFFmpegFrames";
const LEGACY_WIDGET_NAMES = [
    "video",
    "force_rate",
    "custom_width",
    "custom_height",
    "frame_load_cap",
    "skip_first_frames",
    "select_every_nth",
    "format",
];
const VIDEO_ACCEPT = [
    "video/webm",
    "video/mp4",
    "video/x-matroska",
    "video/quicktime",
    "image/gif",
    ".mkv",
    ".mov",
].join(",");


function chainCallback(target, name, callback) {
    const previous = target?.[name];
    target[name] = function (...args) {
        const result = previous?.apply(this, args);
        callback.apply(this, args);
        return result;
    };
}


function fitHeight(node) {
    const computed = node.computeSize?.([node.size[0], node.size[1]]);
    if (computed) node.setSize([node.size[0], computed[1]]);
    node.graph?.setDirtyCanvas?.(true, true);
}


function roundToPrecision(number, precision) {
    const fixed = Number(number).toFixed(precision);
    return fixed.replace(/\.?0+$/, "");
}


function useVhsNumberWidgets(nodeData) {
    const inputs = {
        ...(nodeData?.input?.required || {}),
        ...(nodeData?.input?.optional || {}),
    };
    for (const input of Object.values(inputs)) {
        if (!input || !["INT", "FLOAT"].includes(input[0])) continue;
        input[1] ??= {};
        input[1].widgetType ??= `VHS${input[0]}`;
    }
}


function addNamedWidgetState(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        chainCallback(this, "onConfigure", function (info) {
            if (!this.widgets || !info?.widgets_values ||
                typeof info.widgets_values !== "object") return;

            let widgetValues = info.widgets_values;
            if (Array.isArray(widgetValues)) {
                widgetValues = Object.fromEntries(
                    LEGACY_WIDGET_NAMES
                        .slice(0, widgetValues.length)
                        .map((name, index) => [name, widgetValues[index]]),
                );
            }

            const restoreOrder = [...this.widgets].sort((left, right) => {
                if (left.name === "format") return -1;
                if (right.name === "format") return 1;
                if (left.name === "videopreview") return 1;
                if (right.name === "videopreview") return -1;
                return 0;
            });
            for (const widget of restoreOrder) {
                if (widget.type === "button" ||
                    !Object.prototype.hasOwnProperty.call(widgetValues, widget.name)) {
                    continue;
                }
                const restoredValue = widgetValues[widget.name];
                if (widget.name === "videopreview" && restoredValue &&
                    typeof restoredValue === "object") {
                    const currentValue = widget.value && typeof widget.value === "object"
                        ? widget.value
                        : {};
                    widget.value = {
                        ...currentValue,
                        ...restoredValue,
                        params: {
                            ...(currentValue.params || {}),
                            ...(restoredValue.params || {}),
                        },
                    };
                } else {
                    widget.value = restoredValue;
                }
                widget.callback?.(widget.value);
            }
        });

        chainCallback(this, "onSerialize", function (info) {
            if (!this.widgets) return;
            info.widgets_values = {};
            for (const widget of this.widgets) {
                if (widget.type !== "button") {
                    info.widgets_values[widget.name] = widget.value;
                }
            }
        });
    });
}


function splitInputPath(path) {
    const normalized = String(path || "").replaceAll("\\", "/");
    const separator = normalized.lastIndexOf("/");
    if (separator < 0) return { filename: normalized, subfolder: "" };
    return {
        filename: normalized.slice(separator + 1),
        subfolder: normalized.slice(0, separator),
    };
}


function addLoadFormatBehavior(node, nodeData) {
    const formatWidget = node.widgets?.find((widget) => widget.name === "format");
    const formats = nodeData?.input?.optional?.format?.[1]?.formats;
    if (!formatWidget || !formats) return;
    formatWidget.options.formats = formats;

    const controlledNames = new Set([
        "force_rate",
        "custom_width",
        "custom_height",
        "frame_load_cap",
    ]);
    const baseOptions = new Map();
    for (const widget of node.widgets) {
        if (controlledNames.has(widget.name)) {
            baseOptions.set(widget.name, { ...widget.options });
        }
    }

    const applyFormat = (value) => {
        const source = formats[value] || {};
        const settings = { ...source };
        if (source.target_rate !== undefined) {
            settings.force_rate = { reset: source.target_rate };
        }
        if (source.dim) {
            settings.custom_width = { step: source.dim[0], mod: source.dim[1] };
            settings.custom_height = { step: source.dim[0], mod: source.dim[1] };
            if (source.dim[2]) settings.custom_width.reset = source.dim[2];
            if (source.dim[3]) settings.custom_height.reset = source.dim[3];
        }
        if (source.frames) {
            settings.frame_load_cap = { step: source.frames[0], mod: source.frames[1] };
        }

        for (const widget of node.widgets) {
            if (!baseOptions.has(widget.name)) continue;
            const wasDefault = widget.options?.reset === widget.value;
            widget.options = {
                ...baseOptions.get(widget.name),
                ...(settings[widget.name] || {}),
            };
            if (wasDefault && widget.options.reset !== undefined) {
                widget.value = widget.options.reset;
            }
            widget.callback?.(widget.value);
        }
        node.setDirtyCanvas?.(true, true);
    };

    chainCallback(formatWidget, "callback", applyFormat);

    const capWidget = node.widgets.find((widget) => widget.name === "frame_load_cap");
    if (capWidget) {
        capWidget.annotation = (value) => {
            const maxFrames = node.video_query?.loaded?.frames;
            if (!maxFrames || (value && value < maxFrames)) return;
            const format = formats[formatWidget.value];
            const divisor = format?.frames?.[0] ?? 1;
            const remainder = format?.frames?.[1] ?? 0;
            let loadableFrames = maxFrames;
            if (maxFrames % divisor !== remainder) {
                loadableFrames = Math.floor((maxFrames - remainder) / divisor) * divisor
                    + remainder;
            }
            return `${loadableFrames}\u21FD`;
        };
    }

    const rateWidget = node.widgets.find((widget) => widget.name === "force_rate");
    if (rateWidget) {
        rateWidget.annotation = (value) => {
            if (value === 0 && node.video_query?.source?.fps !== undefined) {
                return `${roundToPrecision(node.video_query.source.fps, 2)}\u21FD`;
            }
        };
    }

    applyFormat(formatWidget.value);
}


function addVaeOutputToggle(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        this.reject_ue_connection = (input) => input?.name === "vae";
    });
    chainCallback(
        nodeType.prototype,
        "onConnectionsChange",
        function (connectionType, slot, connected, linkInfo) {
            if (connectionType !== LiteGraph.INPUT || this.inputs?.[slot]?.type !== "VAE") {
                return;
            }
            const output = this.outputs?.[0];
            if (!output) return;
            const nextType = connected && linkInfo ? "LATENT" : "IMAGE";
            if (output.type !== nextType && output.links?.length) {
                this.disconnectOutput(0);
            }
            output.name = nextType;
            output.type = nextType;
            this.setDirtyCanvas(true, true);
        },
    );
}


function addPreviewOptions(nodeType) {
    chainCallback(nodeType.prototype, "getExtraMenuOptions", function (_, options) {
        if (!Array.isArray(options)) return;
        const previewWidget = this.widgets?.find((widget) => widget.name === "videopreview");
        let params = previewWidget?.value?.params;
        if (previewWidget && !params?.filename) {
            const restoredParams = this.getVideoPreviewParams?.();
            if (restoredParams?.filename) {
                previewWidget.value ??= {};
                previewWidget.value.params = { ...restoredParams };
                params = previewWidget.value.params;
                this.refreshVideoPreview?.();
            }
        }
        if (!previewWidget || !params?.filename) return;

        const previewOptions = [];
        const source = this.video_query?.source;
        if (source?.size && source?.fps !== undefined && source?.frames !== undefined) {
            previewOptions.push({
                content: `${source.size.join("x")}@${source.fps}fps ${source.frames}frames`,
                disabled: true,
            });
        }

        const fullQualityUrl = api.apiURL(
            `/view?${new URLSearchParams(params).toString()}`,
        );
        previewOptions.push(
            {
                content: "Open preview",
                callback: () => window.open(fullQualityUrl, "_blank"),
            },
            {
                content: "Save preview",
                callback: () => {
                    const anchor = document.createElement("a");
                    anchor.href = fullQualityUrl;
                    anchor.download = params.filename;
                    document.body.appendChild(anchor);
                    anchor.click();
                    requestAnimationFrame(() => anchor.remove());
                },
            },
        );

        const pauseLabel = previewWidget.value.paused ? "Resume preview" : "Pause preview";
        previewOptions.push({
            content: pauseLabel,
            callback: () => {
                if (previewWidget.value.paused) {
                    previewWidget.videoEl?.play().catch(() => {});
                } else {
                    previewWidget.videoEl?.pause();
                }
                previewWidget.value.paused = !previewWidget.value.paused;
            },
        });

        const visibilityLabel = previewWidget.value.hidden ? "Show preview" : "Hide preview";
        previewOptions.push({
            content: visibilityLabel,
            callback: () => {
                previewWidget.value.hidden = !previewWidget.value.hidden;
                previewWidget.parentEl.hidden = previewWidget.value.hidden;
                if (previewWidget.value.hidden) {
                    previewWidget.videoEl?.pause();
                } else if (!previewWidget.value.paused) {
                    previewWidget.videoEl?.play().catch(() => {});
                }
                fitHeight(this);
            },
        });

        previewOptions.push({
            content: "Sync preview",
            callback: () => {
                for (const parent of document.getElementsByClassName("vhs_preview")) {
                    for (const child of parent.children) {
                        if (child.tagName === "VIDEO") {
                            child.currentTime = 0;
                            child.play().catch(() => {});
                        } else if (child.tagName === "IMG" && child.src) {
                            child.src = child.src;
                        }
                    }
                }
            },
        });

        const muteLabel = previewWidget.value.muted ? "Unmute Preview" : "Mute Preview";
        previewOptions.push({
            content: muteLabel,
            callback: () => {
                previewWidget.value.muted = !previewWidget.value.muted;
                if (previewWidget.videoEl?.matches(":hover")) {
                    previewWidget.videoEl.muted = previewWidget.value.muted;
                }
            },
        });

        if (options.length && options[0] !== null) previewOptions.push(null);
        options.unshift(...previewOptions);
    });
}


function addUploadAndPreview(nodeType, nodeData) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const node = this;
        const videoWidget = node.widgets?.find((widget) => widget.name === "video");
        if (!videoWidget) return;

        const fileInput = document.createElement("input");
        Object.assign(fileInput, {
            type: "file",
            accept: VIDEO_ACCEPT,
            style: "display: none",
        });
        document.body.appendChild(fileInput);

        const element = document.createElement("div");
        Object.assign(element.style, {
            width: "100%",
            boxSizing: "border-box",
        });
        const previewParent = document.createElement("div");
        previewParent.className = "vhs_preview stdismas_vhs_preview";
        previewParent.style.width = "100%";
        previewParent.hidden = true;

        const video = document.createElement("video");
        video.controls = false;
        video.autoplay = true;
        video.loop = true;
        video.muted = true;
        video.playsInline = true;
        Object.assign(video.style, {
            width: "100%",
            height: "auto",
            display: "block",
        });
        const image = document.createElement("img");
        image.style.width = "100%";
        image.style.display = "block";
        image.hidden = true;
        previewParent.append(video, image);
        element.appendChild(previewParent);

        const fileButton = node.addWidget(
            "button",
            "choose video to upload",
            null,
            () => fileInput.click(),
        );
        fileButton.serialize = false;

        element.value = {
            hidden: false,
            paused: false,
            params: {},
            // Match VHS: previews stay muted outside the node, but are audible
            // while hovered unless the user has explicitly muted previews.
            muted: app.ui.settings.getSettingValue("VHS.DefaultMute"),
        };
        const previewWidget = node.addDOMWidget("videopreview", "preview", element, {
            serialize: false,
            hideOnZoom: false,
            getValue: () => element.value,
            setValue: (value) => {
                element.value = value;
            },
        });
        previewWidget.value = element.value;
        previewWidget.parentEl = previewParent;
        previewWidget.videoEl = video;
        previewWidget.imgEl = image;
        previewWidget.computeSize = function (width) {
            if (this.aspectRatio && !this.parentEl.hidden) {
                const height = Math.max((node.size[0] - 20) / this.aspectRatio + 10, 0);
                this.computedHeight = height + 10;
                return [width, height];
            }
            return [width, -4];
        };

        for (const [eventName, canvasHandler] of [
            ["contextmenu", "_mousedown_callback"],
            ["pointerdown", "_mousedown_callback"],
            ["mousewheel", "_mousewheel_callback"],
            ["pointermove", "_mousemove_callback"],
            ["pointerup", "_mouseup_callback"],
        ]) {
            element.addEventListener(eventName, (event) => {
                event.preventDefault();
                return app.canvas?.[canvasHandler]?.(event);
            }, true);
        }
        element.addEventListener("dragover", (event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
            app.dragOverNode = node;
        });

        video.addEventListener("loadedmetadata", () => {
            previewWidget.aspectRatio = video.videoWidth / video.videoHeight;
            previewParent.hidden = false;
            fitHeight(node);
            if (!previewWidget.value.paused) video.play().catch(() => {});
        });
        video.addEventListener("error", () => {
            previewParent.hidden = true;
            fitHeight(node);
        });
        image.addEventListener("load", () => {
            previewWidget.aspectRatio = image.naturalWidth / image.naturalHeight;
            previewParent.hidden = false;
            fitHeight(node);
        });
        video.addEventListener("mouseenter", () => {
            video.muted = previewWidget.value.muted;
        });
        video.addEventListener("mouseleave", () => {
            video.muted = true;
        });

        let previewTimer = null;
        let queryVersion = 0;
        const previewParams = () => {
            const path = videoWidget.value;
            if (!path) return null;
            const reference = splitInputPath(path);
            const extension = reference.filename.split(".").pop()?.toLowerCase() || "mp4";
            return {
                filename: reference.filename,
                subfolder: reference.subfolder,
                type: "input",
                format: `video/${extension}`,
                force_rate: node.widgets.find((w) => w.name === "force_rate")?.value || 0,
                frame_load_cap:
                    node.widgets.find((w) => w.name === "frame_load_cap")?.value || 0,
                skip_first_frames:
                    node.widgets.find((w) => w.name === "skip_first_frames")?.value || 0,
                select_every_nth:
                    node.widgets.find((w) => w.name === "select_every_nth")?.value || 1,
                custom_width:
                    node.widgets.find((w) => w.name === "custom_width")?.value || 0,
                custom_height:
                    node.widgets.find((w) => w.name === "custom_height")?.value || 0,
            };
        };

        const queryVideo = async (params) => {
            const version = ++queryVersion;
            delete node.video_query;
            node.setDirtyCanvas?.(true, true);
            try {
                const response = await api.fetchApi(
                    `/vhs/queryvideo?${new URLSearchParams(params).toString()}`,
                );
                const query = await response.json();
                if (version !== queryVersion || String(videoWidget.value).replaceAll("\\", "/") !== [
                    params.subfolder,
                    params.filename,
                ].filter(Boolean).join("/")) return;
                node.video_query = query;
                node.setDirtyCanvas?.(true, true);
            } catch (_) {
                // A missing annotation should not prevent the preview itself.
            }
        };

        const updatePreview = async () => {
            const params = previewParams();
            if (!params) {
                queryVersion += 1;
                delete node.video_query;
                video.pause();
                video.removeAttribute("src");
                video.load();
                image.removeAttribute("src");
                previewParent.hidden = true;
                previewWidget.aspectRatio = null;
                fitHeight(node);
                return;
            }
            previewWidget.value.params = { ...params };
            previewParent.hidden = previewWidget.value.hidden;
            queryVideo(params);

            const advancedSetting = app.ui.settings.getSettingValue("VHS.AdvancedPreviews");
            const advancedPreview = advancedSetting !== "Never";
            const extension = params.format.split("/")[1];
            if (!advancedPreview && extension === "gif") {
                image.src = api.apiURL(`/view?${new URLSearchParams({
                    ...params,
                    timestamp: Date.now(),
                }).toString()}`);
                image.hidden = false;
                video.hidden = true;
                return;
            }

            const sourceParams = { ...params, timestamp: Date.now() };
            let endpoint = "/view";
            if (advancedPreview) {
                let targetWidth = Math.max((node.size[0] - 20) * 2, 256);
                const minimumWidth = Number(
                    app.ui.settings.getSettingValue("VHS.AdvancedPreviewsMinWidth") || 0,
                );
                targetWidth = Math.max(targetWidth, minimumWidth);
                if (!params.custom_width || !params.custom_height) {
                    sourceParams.force_size = `${targetWidth}x?`;
                } else {
                    const aspectRatio = params.custom_width / params.custom_height;
                    sourceParams.force_size = `${targetWidth}x${targetWidth / aspectRatio}`;
                }
                sourceParams.deadline = app.ui.settings.getSettingValue(
                    "VHS.AdvancedPreviewsDeadline",
                ) || "realtime";
                endpoint = "/vhs/viewvideo";
            }
            video.autoplay = !previewWidget.value.paused && !previewWidget.value.hidden;
            video.hidden = false;
            image.hidden = true;
            video.src = api.apiURL(
                `${endpoint}?${new URLSearchParams(sourceParams).toString()}`,
            );
            video.load();
            if (video.autoplay) video.play().catch(() => {});
        };
        const schedulePreview = (immediate = false) => {
            if (previewTimer) clearTimeout(previewTimer);
            previewTimer = setTimeout(updatePreview, immediate ? 0 : 150);
        };
        previewWidget.callback = () => schedulePreview(true);
        node.getVideoPreviewParams = previewParams;
        node.refreshVideoPreview = () => schedulePreview(true);

        const addOption = (path) => {
            const values = videoWidget.options?.values;
            if (Array.isArray(values) && !values.includes(path)) values.push(path);
        };
        const choosePath = (path) => {
            addOption(path);
            videoWidget.value = path;
            videoWidget.callback?.(path);
            node.graph?.change?.();
            schedulePreview(true);
        };
        const uploadFile = async (file) => {
            if (!file) return;
            const temporaryUrl = URL.createObjectURL(file);
            previewParent.hidden = false;
            video.hidden = false;
            image.hidden = true;
            video.src = temporaryUrl;
            video.load();
            video.play().catch(() => {});
            try {
                const body = new FormData();
                body.append("image", file);
                body.append("type", "input");
                const response = await api.fetchApi("/upload/image", {
                    method: "POST",
                    body,
                });
                const payload = await response.json();
                if (!response.ok) {
                    throw new Error(payload.error || `Upload failed (${response.status})`);
                }
                const path = [payload.subfolder, payload.name]
                    .filter(Boolean)
                    .join("/")
                    .replaceAll("\\", "/");
                choosePath(path);
            } catch (error) {
                alert(error.message || "Video upload failed");
            } finally {
                URL.revokeObjectURL(temporaryUrl);
                fileInput.value = "";
            }
        };

        fileInput.onchange = () => uploadFile(fileInput.files?.[0]);

        chainCallback(videoWidget, "callback", () => schedulePreview(true));
        for (const name of [
            "force_rate",
            "custom_width",
            "custom_height",
            "frame_load_cap",
            "skip_first_frames",
            "select_every_nth",
        ]) {
            const widget = node.widgets.find((item) => item.name === name);
            if (widget) chainCallback(widget, "callback", () => schedulePreview());
        }

        const previousDragOver = node.onDragOver;
        node.onDragOver = function (event) {
            if (event?.dataTransfer?.files?.length) return true;
            return previousDragOver?.call(this, event);
        };
        const previousDropFile = node.onDropFile;
        node.onDropFile = function (file) {
            const extension = file?.name?.split(".").pop()?.toLowerCase();
            if (["webm", "mp4", "mkv", "gif", "mov"].includes(extension)) {
                uploadFile(file);
                return true;
            }
            return previousDropFile?.call(this, file);
        };

        chainCallback(node, "onRemoved", () => {
            if (previewTimer) clearTimeout(previewTimer);
            queryVersion += 1;
            video.pause();
            fileInput.remove();
        });
        chainCallback(node, "onAdded", () => {
            if (!fileInput.isConnected) document.body.appendChild(fileInput);
            schedulePreview(true);
        });
        chainCallback(node, "onConfigure", () => schedulePreview(true));
        addLoadFormatBehavior(node, nodeData);
        schedulePreview(true);
    });
}


app.registerExtension({
    name: "StDismas.VHSFFmpegFrames",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;
        useVhsNumberWidgets(nodeData);
        addNamedWidgetState(nodeType);
        addVaeOutputToggle(nodeType);
        addPreviewOptions(nodeType);
        addUploadAndPreview(nodeType, nodeData);
    },
});
