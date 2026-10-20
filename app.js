import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "DyPE.MultiAttentionConditioning",
    async nodeCreated(node) {
        if (node.constructor.type !== "MultiAttentionConditioning") return;

        const getIndexFromName = (name) => {
            const parts = name.split('_');
            for (const part of parts) {
                if (/^\d+$/.test(part)) {
                    return parseInt(part, 10);
                }
            }
            return null;
        };

        const updateDynamicFields = (targetCount) => {
            // 1. Clean up inputs (sockets) AND their auto-generated widgets exceeding targetCount
            if (node.inputs) {
                for (let i = node.inputs.length - 1; i >= 0; i--) {
                    const input = node.inputs[i];
                    const idx = getIndexFromName(input.name);
                    if (idx !== null && idx > targetCount) {
                        // If this input spawned an internal widget (like shape 7 strings), clean it from node.widgets first
                        if (input.widget) {
                            const wIndex = node.widgets?.indexOf(input.widget);
                            if (wIndex >= 0) {
                                node.widgets.splice(wIndex, 1);
                            }
                        }
                        node.removeInput(i);
                    }
                }
            }

            // 2. Clean up standalone widgets (UI elements like guidance, blur, resize, strength) exceeding targetCount
            if (node.widgets) {
                for (let i = node.widgets.length - 1; i >= 0; i--) {
                    const idx = getIndexFromName(node.widgets[i].name);
                    if (idx !== null && idx > targetCount) {
                        node.widgets.splice(i, 1);
                    }
                }
            }

            // 3. Add fields in your exact requested order for indices 1 up to targetCount
            for (let i = 1; i <= targetCount; i++) {
                // 1. clip_l wireable string input + text box widget
                const clipFieldName = `clip_l_${i}`;
                if (!node.inputs?.some(inp => inp.name === clipFieldName)) {
                    node.addInput(clipFieldName, "STRING", { shape: 7 });
                }

                // 2. t5xxl wireable string input + text box widget
                const t5FieldName = `t5xxl_${i}`;
                if (!node.inputs?.some(inp => inp.name === t5FieldName)) {
                    node.addInput(t5FieldName, "STRING", { shape: 7 });
                }

                // 3. guidance number widget
                if (!node.widgets?.some(w => w.name === `guidance_${i}`)) {
                    node.addWidget("number", `guidance_${i}`, 1.0, null, { min: 0.0, max: 100.0, step: 0.1, precision: 1 });
                    node.tooltip = "Sets the classifier-free guidance scale for this conditioning.";
                }

                // 4. reference_image input socket
                if (!node.inputs?.some(inp => inp.name === `reference_image_${i}`)) {
                    node.addInput(`reference_image_${i}`, "IMAGE");
                }

                // 5. mask input socket
                if (!node.inputs?.some(inp => inp.name === `mask_${i}`)) {
                    node.addInput(`mask_${i}`, "MASK");
                }

                // 6. mask blur widget
                if (!node.widgets?.some(w => w.name === `mask_${i}_blur`)) {
                    node.addWidget("number", `mask_${i}_blur`, 0, null, { min: 0, max: 256, step: 1, precision: 0 });
                }

                // 7. mask resize widget
                if (!node.widgets?.some(w => w.name === `mask_${i}_resize`)) {
                    node.addWidget("number", `mask_${i}_resize`, 0, null, { min: -256, max: 256, step: 1, precision: 0 });
                }

                // 8. mask strength widget
                if (!node.widgets?.some(w => w.name === `mask_${i}_strength`)) {
                    node.addWidget("number", `mask_${i}_strength`, 1.0, null, { min: 0.0, max: 10.0, step: 0.01, precision: 2 });
                }
            }

            node.setSize(node.computeSize());
            node.setDirtyCanvas(true, true);
        };

        const inputCountWidget = node.widgets.find(w => w.name === "inputcount");
        if (inputCountWidget) {
            // Run once on creation to match current inputcount value
            updateDynamicFields(inputCountWidget.value);

            const originalCallback = inputCountWidget.callback;
            inputCountWidget.callback = function() {
                originalCallback?.apply(this, arguments);
                updateDynamicFields(inputCountWidget.value);
            };
        }
    }
});