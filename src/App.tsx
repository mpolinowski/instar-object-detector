import { useEffect, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { resolve, join } from "@tauri-apps/api/path";
import { LazyLoadImage } from "react-lazy-load-image-component";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ImageEventPayload {
  od_path: string;
  seg_path: string;
  od_time_ms: number;
  seg_time_ms: number;
  video_prep_ms: number;
}

type ImagePayload = {
  bbox: string; // Maps from od_path
  mask: string; // Maps from seg_path
  odTime: number; // Maps from od_time_ms
  segTime: number; // Maps from seg_time_ms
  videoPrepTime: number;
};

export default function App() {
  const [logs, setLogs] = useState<string[]>([]);
  const [images, setImages] = useState<ImagePayload[]>([]);

  // Keep states initially empty while paths resolve asynchronously
  const [inputDir, setInputDir] = useState<string>("");
  const [odModel, setOdModel] = useState<string>("");
  const [segModel, setSegModel] = useState<string>("");
  const [odLabelSet, setOdLabelSet] = useState<string>("auto"); // auto | coco | motion | static | custom
  const [odLabelsFile, setOdLabelsFile] = useState<string>("");
  const [segLabelSet, setSegLabelSet] = useState<string>("auto");
  const [segLabelsFile, setSegLabelsFile] = useState<string>("");
  const [confThreshold, setConfThreshold] = useState<number>(0.45);
  const [isRunning, setIsRunning] = useState<boolean>(false);

  // Auto-suggest a label set from a single model's filename: motion/static
  // dataset models carry their dataset name, anything else is COCO-trained.
  const suggestedLabelFor = (modelPath: string): "coco" | "motion" | "static" => {
    const n = `${modelPath}`.toLowerCase();
    if (n.includes("static")) return "static";
    if (n.includes("motion")) return "motion";
    return "coco";
  };

  const suggestionDisplay = (s: "coco" | "motion" | "static") =>
    ({ coco: "COCO", motion: "Motion", static: "Static" }[s]);

  const baseName = (p: string) => p.split("/").pop() ?? p;

  // Initialize Default Paths on Mount
  useEffect(() => {
    const initializeDefaultPaths = async () => {
      try {
        // Resolve the absolute base execution directory of your app safely
        const baseDir = await resolve(".");

        // Programmatically stitch the absolute paths together
        const defaultInputDir = await join(baseDir, "input_images");
        const defaultOdModel = await join(baseDir, "models", "yolo26n.onnx");
        const defaultSegModel = await join(
          baseDir,
          "models",
          "yolo26n-seg.onnx",
        );

        setInputDir(defaultInputDir);
        setOdModel(defaultOdModel);
        setSegModel(defaultSegModel);
      } catch (error) {
        console.error("Failed to map system fallback directory tracks:", error);
      }
    };

    initializeDefaultPaths();
  }, []);

  // Event Listeners Setup Block
  useEffect(() => {
    let unlistenLogs: () => void;
    let unlistenImages: () => void;

    const setupListeners = async () => {
      unlistenLogs = await listen<string>("log_event", (event) => {
        setLogs((prev) => [...prev, event.payload]);
      });

      unlistenImages = await listen<ImageEventPayload>(
        "image_event",
        (event) => {
          const { od_path, seg_path, od_time_ms, seg_time_ms, video_prep_ms } =
            event.payload;

          // Transform and Map keys
          const processed: ImagePayload = {
            bbox: convertFileSrc(od_path),
            mask: convertFileSrc(seg_path),
            odTime: od_time_ms,
            segTime: seg_time_ms,
            videoPrepTime: video_prep_ms,
          };

          setImages((prev) => [...prev, processed]);
        },
      );
    };

    setupListeners();
    return () => {
      if (unlistenLogs) unlistenLogs();
      if (unlistenImages) unlistenImages();
    };
  }, []);

  const selectFolder = async () => {
    const selected = await open({
      directory: true,
      multiple: false,
      defaultPath: inputDir || undefined,
    });
    if (typeof selected === "string") setInputDir(selected);
  };

  const selectOdModel = async () => {
    const selected = await open({
      multiple: false,
      defaultPath: odModel || undefined,
      filters: [{ name: "ONNX Model", extensions: ["onnx"] }],
    });
    if (typeof selected === "string") setOdModel(selected);
  };

  const selectSegModel = async () => {
    const selected = await open({
      multiple: false,
      defaultPath: segModel || undefined,
      filters: [{ name: "ONNX Model", extensions: ["onnx"] }],
    });
    if (typeof selected === "string") setSegModel(selected);
  };

  const selectLabelsFile = async (
    defaultPath: string,
    apply: (path: string) => void,
  ) => {
    const selected = await open({
      multiple: false,
      defaultPath: defaultPath || undefined,
      filters: [
        { name: "Label file", extensions: ["txt", "yaml", "yml", "names"] },
      ],
    });
    if (typeof selected === "string") apply(selected);
  };

  const executeInference = async () => {
    setLogs([]);
    setImages([]);
    setIsRunning(true);

    try {
      const labelArg = (sel: string, file: string) =>
        sel === "auto" ? "auto" : sel === "custom" ? file : sel;

      await invoke("run_inference", {
        inputDir,
        odModelPath: odModel,
        segModelPath: segModel,
        confThreshold,
        odLabels: labelArg(odLabelSet, odLabelsFile),
        segLabels: labelArg(segLabelSet, segLabelsFile),
      });
    } catch (err) {
      setLogs((prev) => [...prev, `Runtime Error: ${err}`]);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <main className="p-6 space-y-6 max-w-6xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 border p-4 rounded-xl bg-zinc-50">
        <div className="space-y-1">
          <Button variant="outline" className="w-full" onClick={selectFolder}>
            {inputDir ? "✓ Target Path Configured" : "1. Choose Images Target"}
          </Button>
          <p className="text-xs text-zinc-500 truncate px-1">
            {inputDir || "Resolving directory tracks..."}
          </p>
        </div>

        <div className="space-y-1">
          <Button variant="outline" className="w-full" onClick={selectOdModel}>
            {odModel
              ? "✓ OD Model Target Linked"
              : "2. Select OD Model (.onnx)"}
          </Button>
          <p className="text-xs text-zinc-500 truncate px-1">
            {odModel || "Resolving model tracks..."}
          </p>
        </div>

        <div className="space-y-1">
          <Button variant="outline" className="w-full" onClick={selectSegModel}>
            {segModel
              ? "✓ SEG Model Target Linked"
              : "3. Select SEG Model (.onnx)"}
          </Button>
          <p className="text-xs text-zinc-500 truncate px-1">
            {segModel || "Resolving model tracks..."}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 border p-4 rounded-xl bg-zinc-50 md:grid-cols-2">
        {/* Object-detection label set */}
        <div className="space-y-1">
          <label htmlFor="od-label-set" className="text-xs font-bold text-zinc-700">
            4a. OD label set — {baseName(odModel)}
          </label>
          <select
            id="od-label-set"
            className="w-full max-w-sm border border-zinc-300 rounded-md px-3 py-2 text-sm bg-white"
            value={odLabelSet}
            onChange={(e) => setOdLabelSet(e.target.value)}
          >
            <option value="auto">
              Auto — suggested: {suggestionDisplay(suggestedLabelFor(odModel))}
            </option>
            <option value="coco">COCO (80 classes)</option>
            <option value="motion">Motion (19 classes)</option>
            <option value="static">Static (26 classes)</option>
            <option value="custom">Custom label file...</option>
          </select>
          <p className="text-xs text-zinc-500">
            {odLabelSet === "custom"
              ? odLabelsFile
                ? `✓ ${odLabelsFile}`
                : "Choose the class list file for the OD model."
              : odLabelSet === "auto"
                ? `Will use ${suggestionDisplay(suggestedLabelFor(odModel))} classes, suggested from the model name.`
                : "Built-in table — locked to the dataset."}
          </p>
          {odLabelSet === "custom" && (
            <div className="flex items-center gap-3 pt-1">
              <Button
                variant="outline"
                onClick={() =>
                  selectLabelsFile(odLabelsFile, setOdLabelsFile)
                }
                className="text-xs"
              >
                Choose OD labels file (.txt / .yaml)
              </Button>
              <span className="text-xs text-zinc-500">
                One name per line,{" "}
                <code className="font-mono">names: ['a', 'b']</code> or{" "}
                <code className="font-mono">0: name</code> YAML.
              </span>
            </div>
          )}
        </div>

        {/* Segmentation label set */}
        <div className="space-y-1">
          <label htmlFor="seg-label-set" className="text-xs font-bold text-zinc-700">
            4b. SEG label set — {baseName(segModel)}
          </label>
          <select
            id="seg-label-set"
            className="w-full max-w-sm border border-zinc-300 rounded-md px-3 py-2 text-sm bg-white"
            value={segLabelSet}
            onChange={(e) => setSegLabelSet(e.target.value)}
          >
            <option value="auto">
              Auto — suggested: {suggestionDisplay(suggestedLabelFor(segModel))}
            </option>
            <option value="coco">COCO (80 classes)</option>
            <option value="motion">Motion (19 classes)</option>
            <option value="static">Static (26 classes)</option>
            <option value="custom">Custom label file...</option>
          </select>
          <p className="text-xs text-zinc-500">
            {segLabelSet === "custom"
              ? segLabelsFile
                ? `✓ ${segLabelsFile}`
                : "Choose the class list file for the SEG model."
              : segLabelSet === "auto"
                ? `Will use ${suggestionDisplay(suggestedLabelFor(segModel))} classes, suggested from the model name.`
                : "Built-in table — locked to the dataset."}
          </p>
          {segLabelSet === "custom" && (
            <div className="flex items-center gap-3 pt-1">
              <Button
                variant="outline"
                onClick={() =>
                  selectLabelsFile(segLabelsFile, setSegLabelsFile)
                }
                className="text-xs"
              >
                Choose SEG labels file (.txt / .yaml)
              </Button>
              <span className="text-xs text-zinc-500">
                One name per line,{" "}
                <code className="font-mono">names: ['a', 'b']</code> or{" "}
                <code className="font-mono">0: name</code> YAML.
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-4 border p-4 rounded-xl bg-zinc-50 items-center">
        <div className="w-full md:w-72 space-y-1">
          <label className="text-xs font-bold text-zinc-700">
            5. Confidence threshold:{" "}
            <span className="font-mono tabular-nums">
              {confThreshold.toFixed(2)}
            </span>
          </label>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.05}
            value={confThreshold}
            onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
            className="w-full accent-emerald-600"
            aria-label="Confidence threshold"
          />
          <p className="text-xs text-zinc-500">
            Detections below this score are discarded after NMS.
          </p>
        </div>
      </div>

      <Button
        className="w-full h-12 text-md font-semibold"
        onClick={executeInference}
        disabled={
          isRunning ||
          !inputDir ||
          !odModel ||
          !segModel ||
          (odLabelSet === "custom" && !odLabelsFile) ||
          (segLabelSet === "custom" && !segLabelsFile)
        }
      >
        {isRunning
          ? "Running Neural Evaluation..."
          : "Begin Pipeline Inference"}
      </Button>

      <div className="space-y-6">
        {images.map((item, idx) => (
          <Card key={idx} className="py-0 gap-0 shadow-sm border border-zinc-100">
            <CardHeader className="py-3 px-5 border-b border-zinc-700 bg-zinc-950 flex flex-row items-center justify-between space-y-0">
              <CardTitle className="text-xs font-mono text-zinc-300">
                Pipeline Record:{" "}
                <span className="text-white font-bold">#{idx + 1}</span>
              </CardTitle>
              {/* Performance Badge Row */}
              <div className="flex gap-4 text-[11px] font-mono">
                {item.videoPrepTime > 0 && (
                  <span className="text-blue-400">
                    Video Prep:{" "}
                    <strong className="font-bold">
                      {item.videoPrepTime}ms
                    </strong>
                  </span>
                )}
                <span className="text-amber-400">
                  OD Core:{" "}
                  <strong className="font-bold">{item.odTime}ms</strong>
                </span>
                <span className="text-cyan-400">
                  SEG Core:{" "}
                  <strong className="font-bold">{item.segTime}ms</strong>
                </span>
                <span className="text-zinc-400">
                  Combined Matrix:{" "}
                  <strong className="text-zinc-100 font-bold">
                    {item.videoPrepTime + item.odTime + item.segTime}ms
                  </strong>
                </span>
              </div>
            </CardHeader>
            <CardContent className="p-4">
              <div className="text-xs font-mono text-zinc-400 mb-3 border-b pb-1">
                Batch Sequence Processing Index: {idx + 1}
              </div>
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-1 text-center">
                  <span className="text-xs font-bold text-zinc-700 block">
                    Object Bounding Box Topology
                  </span>
                  <LazyLoadImage
                    src={item.bbox}
                    effect="blur"
                    className="rounded-lg border bg-zinc-900 object-contain w-full max-h-[450px] transition-transform duration-300 group-hover:scale-[1.01]"
                  />
                </div>
                <div className="space-y-1 text-center">
                  <span className="text-xs font-bold text-zinc-700 block">
                    Neural Semantic Segmentation Mask
                  </span>
                  <LazyLoadImage
                    src={item.mask}
                    effect="blur"
                    className="rounded-lg border bg-zinc-900 object-contain w-full max-h-[450px] transition-transform duration-300 group-hover:scale-[1.01]"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="py-0 gap-0 shadow-sm">
        <CardContent className="p-4">
          <ScrollArea className="h-44 w-full rounded border p-4 bg-black text-emerald-400 font-mono text-xs leading-relaxed">
            {logs.length === 0 && (
              <span className="text-zinc-600">Console Pipeline Idle...</span>
            )}
            {logs.map((log, index) => (
              <div key={index}>{log}</div>
            ))}
          </ScrollArea>
        </CardContent>
      </Card>
    </main>
  );
}
