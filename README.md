# Virtual-Assistant-Pipeline

This repository contains a Python-based evaluation pipeline for intent recognition using a large language model (LLM) from Hugging Face. It is designed to be lightweight on GitHub, with heavy models downloaded at runtime.

---

## 1. Requirements

- Python 3.11+  
- Git  
- Virtual environment (venv recommended)  

Dependencies are listed in \`requirements.txt\`:

\`\`\`
torch>=2.2.0
torchvision>=0.17.0
numpy==1.26.0
Pillow==10.2.0
transformers>=4.45.0
\`\`\`

> These versions ensure compatibility between PyTorch and Hugging Face Transformers.

---

## 2. Setup

### Step 1: Clone the repository

\`\`\`bash
git clone https://github.com/sam88aalap/Virtual-Assistant-Pipeline.git
cd Virtual-Assistant-Pipeline
\`\`\`

### Step 2: Create a virtual environment

\`\`\`bash
python -m venv venv
\`\`\`

### Step 3: Activate the virtual environment

- **Windows PowerShell / Git Bash**:

\`\`\`bash
venv\Scripts\activate
\`\`\`

- **Linux / macOS**:

\`\`\`bash
source venv/bin/activate
\`\`\`

### Step 4: Install dependencies

\`\`\`bash
pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

---

## 3. Downloading Models

This repo uses Hugging Face models which are downloaded automatically at runtime.  

- Model: \`microsoft/Phi-3-mini-4k-instruct\`  
- Stored locally in the PyTorch cache (\`~/.cache/torch\` by default).  

Optionally, you can pre-download models:

\`\`\`bash
python download_models.py
\`\`\`

---

## 4. Running Evaluation

Run the intent evaluation script:

\`\`\`bash
python evaluation/evaluate_intent.py
\`\`\`

- Uses \`evaluation/intent_data.json\` as input.  
- Output will be displayed in the console.  
- Automatically uses GPU if available (\`DEVICE = "cuda"\`), otherwise CPU.

---

## 5. Repository Structure

\`\`\`
Virtual-Assistant-Pipeline/
│
├─ download_models.py      # Script to pre-download models
├─ evaluation/             # Evaluation scripts
│   ├─ evaluate_intent.py
│   └─ intent_data.json
├─ requirements.t
