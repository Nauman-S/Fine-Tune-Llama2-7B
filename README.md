# Fine-Tune-Llama2-7B

## Project Goal
Fine-tune LLaMA-2-7B using Parameter-Efficient Fine-Tuning (PEFT/LoRA) on the Dolly-15K dataset for instruction-following capabilities.

## Dataset: Dolly-15K

**Source**: [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

### Dataset Overview
- **Size**: 15,011 instruction-following examples
- **Format**: Each entry contains `instruction`, `context`, and `response` fields
- **Quality**: High-quality, human-generated instruction-response pairs
- **Categories**: 7 different categories of tasks (creative writing, information extraction, etc.)

### Dataset Structure
```json
{
  "instruction": "What is the capital of France?",
  "context": "Geography question about European capitals",
  "response": "The capital of France is Paris."
}
```


### 3. Data Splits
- **Training**: 80% (12,009 examples)
- **Validation**: 10% (1,501 examples)  
- **Test**: 10% (1,502 examples)
- **Access Link**: https://drive.google.com/drive/folders/1CXJHPZEYk-XOypqvg71-j4fkdEvkboct?usp=sharing

### 4. Tokenization & Encoding
- Use LLaMA-2 tokenizer for consistent tokenization
- Handle special tokens and padding appropriately
- Implement proper truncation for long sequences

## Fine-tuning Objective
Transform LLaMA-2-7B from a general language model to a helpful, conversational assistant that:
- Responds concisely to user instructions
- Follows the specified format consistently
- Avoids generating unnecessary tokens
- Maintains conversational quality