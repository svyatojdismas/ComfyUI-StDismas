import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "stdismas.multi_string_selector",

  nodeCreated(node) {
    if (node.comfyClass !== "MultiStringSelector_StDismas") return;

    const getHidden = () => node.widgets?.find(w => w.name === "values_json");
    const hidden = getHidden();

    // Hide values_json widget (kept for backend)
    if (hidden) {
      hidden.type = "hidden";
      hidden.computeSize = () => [0, 0];
    }

    let values = [];
    try {
      if (hidden?.value) {
        const parsed = JSON.parse(hidden.value);
        if (Array.isArray(parsed)) values = parsed.map(v => String(v));
      }
    } catch (e) {}

    const stringWidgets = [];

    const syncHidden = () => {
      const h = getHidden();
      if (h) h.value = JSON.stringify(values);
      app.graph.setDirtyCanvas(true, true);
    };

    const rebuild = () => {
      // Remove old widgets
      for (const w of stringWidgets) {
        const idx = node.widgets.indexOf(w);
        if (idx >= 0) node.widgets.splice(idx, 1);
      }
      stringWidgets.length = 0;

      // Add current value widgets
      values.forEach((val, i) => {
        const w = node.addWidget(
          "string",
          `string_${i + 1}`,
          val,
          (v) => {
            values[i] = String(v ?? "");
            syncHidden();
          },
          { multiline: false }
        );
        stringWidgets.push(w);
      });

      syncHidden();
    };

    // Controls
    node.addWidget("button", "Add", "+", () => {
      values.push("");
      rebuild();
    });

    node.addWidget("button", "Remove", "-", () => {
      if (values.length > 0) values.pop();
      rebuild();
    });

    node.addWidget("button", "Clear", "x", () => {
      values = [];
      rebuild();
    });

    // Initial build
    rebuild();
  },
});
