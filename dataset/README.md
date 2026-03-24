## Datasets

### Dataset 1 — `claudios/VulDeePecker`
**Link:** https://huggingface.co/datasets/claudios/VulDeePecker

| Attribute | Details |
|---|---|
| **Total Samples** | ~160,000 rows |
| **Train / Val / Test** | 128k / 16k / 16k |
| **Language** | C/C++ |
| **Label** | `0` = Non-vulnerable, `1` = Vulnerable |
| **Bug Types** | CWE-119 (Buffer Overflow), CWE-399 (Resource Management) |
| **Format** | Parquet |
| **Source** | Real-world CVEs from Firefox, FFmpeg, Linux Kernel, etc. |

**Main columns:**

| Column | Description |
|---|---|
| `functionSource` | C/C++ function source code |
| `label` | 0 or 1 (binary classification) |
| `cwe` | Vulnerability type (CWE-119, CWE-399) |
| `vulLine` | Line containing the vulnerability |
| `fName` | Original file path |

**Load dataset:**
```python
from datasets import load_dataset
ds = load_dataset("claudios/VulDeePecker")
```

---

### Dataset 2 — `google/code_x_glue_cc_defect_detection`
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

---

### Combining Both Datasets

To increase training diversity, both datasets will be merged into a unified format:

```python
from datasets import load_dataset, concatenate_datasets

ds1 = load_dataset("claudios/VulDeePecker", split="train")
ds2 = load_dataset("google/code_x_glue_cc_defect_detection", split="train")

# Normalize to common format
ds1 = ds1.map(lambda x: {"code": x["functionSource"], "label": x["label"]})
ds2 = ds2.map(lambda x: {"code": x["func"], "label": int(x["target"])})

# Keep only the 2 required columns
ds1 = ds1.select_columns(["code", "label"])
ds2 = ds2.select_columns(["code", "label"])

# Merge
merged = concatenate_datasets([ds1, ds2])
print(merged)  # ~149,000 samples
```
