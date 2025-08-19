# PatentProSum: Hybrid Extractive-Abstractive Patent Summarization

PatentProSum is a toolkit and research project for generating high-quality abstractive summaries of patent documents. It combines extractive sentence selection (LexRank) with a BART-based language model fine-tuned using Low-Rank Adaptation (LoRA). The project includes training scripts, evaluation notebooks, and user/admin interfaces for summarization.

## Features

- **Hybrid Summarization:** LexRank for extractive selection, BART-LoRA for abstractive generation.
- **Domain Generalization:** Models for general patents and for Textiles/Paper domains.
- **LoRA Fine-Tuning:** Efficient adaptation of large language models.
- **User/Admin UIs:** Interactive summarization interfaces.
- **Evaluation Notebooks:** Training and evaluation scripts/notebooks.

## Models

Two LoRA-adapted BART models are available on Hugging Face:

- **Specialized Model (Textiles & Paper):** [LexBartLo_1 (Exp16)](https://huggingface.co/Nevidu/LexBartLo_1)
- **Generalized Model:** [LexBartLo_2 (Exp17)](https://huggingface.co/Nevidu/LexBartLo_2)

## Getting Started

### Installation

1. Clone this repository.
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

Use the user interface in `User_UI.py` to summarize patent documents. Example:

```python
from User_UI import summarize

summary = summarize(
    text="Your patent description here.",
    max_tokens=256,
    model_type="Generalized"  # or "Specialized (Textiles and Paper)"
)
print(summary)
```

### Model Loading

Models are loaded using Hugging Face's `transformers` and `peft` libraries. See `User_UI.py` for details.

## Project Structure

- `Admin_UI.py`, `User_UI.py`: User/admin interfaces.
- `Notebooks/`: Jupyter notebooks for training, evaluation, and data preparation.
- `M2L_LR_S2_EXT4_EXP16_model/`, `M2L_LR_S2_EXT4_EXP17_model/`: Local model adapters (see Hugging Face links above).
- `requirements.txt`: Python dependencies.

## Dataset

The dataset for training/evaluation is available [here](https://drive.google.com/drive/folders/1PLzZZ6JaHYl5jjp8EEm8GcNVQ1Bb_8cw?usp=sharing).

## Citation

If you use this project or models, please cite:

```
@inproceedings{jayatilleke2025hybrid,
  title={A Hybrid Architecture with Efficient Fine Tuning for Abstractive Patent Document Summarization},
  author={Jayatilleke, Nevidu and Weerasinghe, Ruvan},
  booktitle={2025 International Research Conference on Smart Computing and Systems Engineering (SCSE)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```

## License

[Specify your license here]

## Contact

For questions or contributions, please contact [your email or GitHub profile].
