## Datasets

### Dataset 1 — `google/code_x_glue_cc_defect_detection`
**Link:** https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection

| Attribute | Details |
|---|---|
| **Total Samples** | ~27,300 rows |
| **Train / Val / Test** | 21,854 / 2,732 / 2,732 |
| **Language** | C/C++ |
| **Label** | `False` = Safe, `True` = Defective |
| **Bug Types** | Resource leaks, Use-after-free, DoS, Buffer overflow |
| **Format** | Parquet |
| **Source** | FFmpeg and QEMU — real-world open source projects |
| **License** | C-UDA |

**Main columns:**

| Column | Description |
|---|---|
| `func` | C/C++ function source code |
| `target` | False or True (binary classification) |
| `project` | Project name (FFmpeg / QEMU) |
| `commit_id` | Git commit hash of the bug fix |

**Load dataset:**
```python
from datasets import load_dataset
ds = load_dataset("google/code_x_glue_cc_defect_detection")
```

