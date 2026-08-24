import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


const NODE_NAME = "StDismas_LoadVideoFFmpegFrames";
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
    };

    chainCallback(formatWidget, "callback", applyFormat);
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

        const container = document.createElement("div");
        Object.assign(container.style, {
            width: "100%",
            minHeight: "150px",
            display: "flex",
            flexDirection: "column",
            gap: "6px",
            boxSizing: "border-box",
            padding: "4px",
        });
        const video = document.createElement("video");
        video.controls = true;
        video.loop = true;
        video.muted = true;
        video.playsInline = true;
        Object.assign(video.style, {
            width: "100%",
            minHeight: "120px",
            maxHeight: "360px",
            objectFit: "contain",
            background: "#111",
            borderRadius: "4px",
        });
        const status = document.createElement("div");
        status.textContent = "Choose or drop a video";
        Object.assign(status.style, {
            color: "var(--descrip-text, #aaa)",
            fontSize: "11px",
            textAlign: "center",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
        });
        container.append(video, status);
        node.addDOMWidget("videopreview", "video", container, {
            serialize: false,
            hideOnZoom: false,
            getMinHeight: () => 170,
        });

        let previewTimer = null;
        const previewParams = () => {
            const path = videoWidget.value;
            if (!path) return null;
            const reference = splitInputPath(path);
            const extension = reference.filename.split(".").pop()?.toLowerCase() || "mp4";
            const params = new URLSearchParams({
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
                deadline: "realtime",
                timestamp: Date.now(),
            });
            const width = Number(
                node.widgets.find((w) => w.name === "custom_width")?.value || 0,
            );
            const height = Number(
                node.widgets.find((w) => w.name === "custom_height")?.value || 0,
            );
            const targetWidth = Math.max(Math.round((node.size?.[0] || 256) * 2), 256);
            if (width > 0 && height > 0) {
                params.set("force_size", `${targetWidth}x${Math.round(targetWidth / (width / height))}`);
            } else {
                params.set("force_size", `${targetWidth}x?`);
            }
            return params;
        };

        const updatePreview = () => {
            const params = previewParams();
            if (!params) {
                video.pause();
                video.removeAttribute("src");
                video.load();
                status.textContent = "Choose or drop a video";
                return;
            }
            status.textContent = videoWidget.value;
            video.src = api.apiURL(`/vhs/viewvideo?${params.toString()}`);
            video.load();
        };
        const schedulePreview = (immediate = false) => {
            if (previewTimer) clearTimeout(previewTimer);
            previewTimer = setTimeout(updatePreview, immediate ? 0 : 150);
        };

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
            status.textContent = `Uploading ${file.name}…`;
            const temporaryUrl = URL.createObjectURL(file);
            video.src = temporaryUrl;
            video.load();
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
                status.textContent = error.message || "Video upload failed";
                alert(status.textContent);
            } finally {
                URL.revokeObjectURL(temporaryUrl);
                fileInput.value = "";
            }
        };

        fileInput.onchange = () => uploadFile(fileInput.files?.[0]);
        const uploadWidget = node.addWidget(
            "button",
            "choose video to upload",
            null,
            () => fileInput.click(),
        );
        uploadWidget.serialize = false;

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
            video.pause();
            fileInput.remove();
        });
        addLoadFormatBehavior(node, nodeData);
        schedulePreview(true);
        const minSize = node.computeSize?.() || node.size;
        node.setSize([Math.max(node.size[0], 300), Math.max(minSize[1], 430)]);
    });
}


app.registerExtension({
    name: "StDismas.VHSFFmpegFrames",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;
        addVaeOutputToggle(nodeType);
        addUploadAndPreview(nodeType, nodeData);
    },
});
