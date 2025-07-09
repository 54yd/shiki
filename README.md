# Shiki – AutoCrimeDetector 🕵️‍♀️🔍

An **automated crime-detection** app that analyzes CCTV screenshots (or any image), pinpoints suspects or objects of interest, and returns their normalized coordinates. Built on top of ShowUI (a vision-language model) and Qwen2-VL, it exposes a simple Gradio web UI so you can try it out in seconds.

---

## 🚀 Features

- **AI-Powered Detection**  
  Leverages `showlab/ShowUI-2B` + `Qwen/Qwen2-VL-2B-Instruct` to understand and localize elements in an image.

- **Coordinate Extraction**  
  Returns relative `[x, y]` coordinates (0–1) for any queried element.

- **Gradio Interface**  
  Spin up a local web app (`http://localhost:7860/`) to upload images and type your query.

- **Easy Model Management**  
  Automatically downloads required weights from Hugging Face Hub if not present.

---

## 📁 Repository Structure

```
shiki/
├── client/
│   ├── showui_gradio.py      # Gradio demo for ShowUI + Qwen2-VL inference
│   └── …                     # (other client helpers)
├── env/
│   └── environment.yml       # Conda environment spec
├── create_structure.sh       # Boilerplate setup script
├── .gitignore
└── README.md                 # ← you are here
```

---

## ⚙️ Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/54yd/shiki.git
cd shiki
```

### 2. Create & activate your Conda env
```bash
conda env create -f env/environment.yml
conda activate shiki
```

If you don’t use Conda, install dependencies manually:
```bash
pip install torch torchvision \
            transformers \
            huggingface_hub \
            gradio \
            pillow \
            numpy
```

### 3. Run the Gradio demo
```bash
python client/showui_gradio.py
```

- **Server URL:** http://localhost:7860  
- **Upload your image**, enter a natural-language query (e.g.  
  _“Locate the person in a red jacket”_), and click **Submit**.  
- The model will return a JSON-style reply with coordinates.

---

## 🔍 How It Works

1. **Image upload → NumPy array**  
   The Gradio `Image` input gives you a NumPy array.

2. **Save & preprocess**  
   `showui_gradio.py` converts it to a temporary PNG on disk, then uses:
   - `AutoProcessor` (from `transformers`) to build a message bundle
   - `process_vision_info()` (from `qwen_vl_utils`) to extract image tensors

3. **Model inference**  
   - **ShowUI** (vision-language model) loaded via  
     `Qwen2VLForConditionalGeneration.from_pretrained(...)`
   - Generates text output describing coordinates, e.g.  
     ```
     [{"element":"suspect","coord":[0.45,0.72]}]
     ```

4. **Display result**  
   The Gradio app shows the raw text. You can parse it client-side to draw boxes or trigger actions.

---

## ⚙️ Configuration & Constants

Inside `client/showui_gradio.py` you’ll find configurable values:

| Name            | Default                              | Description                                |
|-----------------|--------------------------------------|--------------------------------------------|
| `MODEL_ID`      | `"showlab/ShowUI-2B"`                | Hugging Face repo for ShowUI weights       |
| `PROCESSOR_ID`  | `"Qwen/Qwen2-VL-2B-Instruct"`        | HF ID for the vision-language processor     |
| `LOCAL_DIR`     | `"./showui-2b"`                      | Local directory to cache downloaded models |
| `MIN_PIXELS`    | `256 * 28 * 28`                      | Minimum pixel count for model input images |
| `MAX_PIXELS`    | `1344 * 28 * 28`                     | Maximum pixel count for model input images |
| `MAX_NEW_TOKENS`| `128`                                | Max tokens to generate per query           |

---

## 🚧 Troubleshooting

- **“Model folder already exists”**  
  Means weights are in `./showui-2b/`. To force a fresh download, delete that folder.

- **Memory / dtype issues**  
  If you have a GPU, consider changing `torch_dtype=torch.float16` and `device_map="auto"` in `load_model_and_processor()`.

- **Slow startup**  
  The first run downloads ~3–4 GB of weights. After that, it’s cached locally.

---

## 🛠️ Next Steps

- **Integrate with your CCTV pipeline**  
  Replace the Gradio wrapper with a REST API or background worker.

- **Parse & visualize**  
  Use the returned coordinates to draw bounding boxes on your images.

- **Custom prompts**  
  Tweak `_SYSTEM` and user messages in `predict()` to fine-tune detection.

---

## 📜 License & Credits

- **Model weights**:  
  – ShowUI by ShowLab ([ShowUI-2B](https://huggingface.co/showlab/ShowUI-2B))  
  – Qwen2-VL by Qwen Team ([Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct))

- **Code**: MIT-licensed (or specify your license here)

---

Happy automating! 🚀👮‍♂️👁️
