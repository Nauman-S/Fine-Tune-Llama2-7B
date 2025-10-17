# Instruction Fine-Tuning of LLaMA-2-7B with PEFT (LoRA) on Dolly-15K

## 🎯 Objective
Fine-tune the **meta-llama/Llama-2-7b** model using **Parameter-Efficient Fine-Tuning (PEFT, via LoRA)** on the **databricks/databricks-dolly-15k** dataset.  
Demonstrate **loss convergence** and show measurable improvement on **AlpacaEval 2** and **MT-Bench** benchmarks compared to the base model.

---

## 🧠 Background
Instruction fine-tuning adapts a pretrained LLM to **follow natural-language instructions**.  
Instead of generic next-token prediction, the model learns to respond *helpfully and coherently* to user prompts.  
This converts a raw pretrained model into a usable conversational assistant.

**Reference Repositories**
- **Stanford Alpaca:** https://github.com/tatsu-lab/stanford_alpaca  
- **Hugging Face PEFT (LoRA):** https://github.com/huggingface/peft  
- **AlpacaEval 2:** https://github.com/tatsu-lab/alpaca_eval  
- **FastChat / MT-Bench:** https://github.com/lm-sys/FastChat  

---

## 🧩 Tasks

### 1. Dataset
- Use **`databricks/databricks-dolly-15k`** (Hugging Face).  
  Each entry: `instruction`, `context`, `response`.
- **Preprocessing**
  - Remove empty `response` entries.  
  - Format as:
    ```
    ### Instruction:
    {instruction}

    ### Context:
    {context}

    ### Response:
    {response}
    ```
  - Split into **train (80%) / validation (10%) / test (10%)**.

### 2. Model & Training
- Base: `meta-llama/Llama-2-7b-hf`
- Use **LoRA** with **PEFT**
- Track **training and validation loss** each epoch.  
- Expect smooth convergence (loss decreasing steadily).  
- Use FSDP if VRAM is restricted

### 3. Evaluation
- Benchmarks:
  - **AlpacaEval 2** → instruction-following quality  
  - **MT-Bench (FastChat)** → multi-turn dialogue quality  

---

## 🧾 Submission

### Deliverables
- **Report (2–4 pages, PDF)** containing:
  1. **Dataset** – source, preprocessing, splits  
  2. **Implementation** – model setup, LoRA config, hyperparams, libraries  
  3. **Evaluation** – benchmarks, metrics, comparison vs. base  
  4. **Hardware** – GPU type, VRAM, runtime  
  5. **Results** – tables + **training / validation loss curves**  
  6. **Discussion / Conclusion** – insights, limitations, next steps  
- **Code / notebook** with reproducible environment (`requirements.txt` or `.yml`).

---

## ✅ Expected Outcomes
- Training loss shows clear convergence.  
- **AlpacaEval 2 / MT-Bench** show positive improvement.  
- No significant degradation in reasoning or factual consistency.  
- Report includes clear loss plots and concise analysis.

---

## 🌟 Bonus Marks (up to +10%)
1. **Distributed Training**
   - Implement **FSDP**, **ZeRO**, **DDP**, or other multi-GPU methods.  
   - Demonstrate correct scaling or efficiency.  
2. **Extended Evaluation**
   - Add more benchmarks (e.g. **MMLU**, **GSM8K**, **BBH**, **ARC**, **TruthfulQA**).  
   - Discuss results — performance drops are acceptable if analyzed clearly.

---
