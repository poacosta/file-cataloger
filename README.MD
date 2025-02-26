# 📂 File Cataloger: Path Pattern Explorer

> Unearth the hidden patterns in your massive directory structures and prepare datasets for meaningful exploration.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ever stared into the abyss of a filesystem with 500,000+ files and wondered "what's actually in here?" File Cataloger
transforms chaotic directory structures into clean, analyzable datasets that reveal the underlying patterns in your
digital organization.

## 🔍 The Dataset Preparation Journey

This tool isn't about indexing for its own sake – it's about creating a foundation for exploration and insight:

1. **Harvest**: Efficiently extract filename and path data from massive directory structures
2. **Transform**: Convert hierarchical file structures into a flat, analyzable CSV format
3. **Explore**: Enable pattern discovery across directories, naming conventions, and file distributions
4. **Insight**: Turn raw file system data into actionable organization intelligence

## 🚀 Features That Actually Matter

- **Scales Like a Champion**: Effortlessly handles 500k+ files across 1M+ directories
- **Respects Your RAM**: Memory-efficient batch processing prevents your system from melting down
- **Speed Without Sacrifice**: Parallel processing gets results faster without trading off reliability
- **Fault Forgiveness**: Gracefully skips permission-restricted directories instead of crashing
- **Focus Flexibility**: Filter for specific file types or catalog everything – you decide

## 🧰 The Technical Bits

### Requirements

- Python 3.10+
- pandas (for data transformation)
- tqdm (for sanity-preserving progress bars)

### Setup

```bash
# Clone the repository
git clone https://github.com/poacosta/file-cataloger.git
cd file-cataloger

# Install dependencies
pip install -r requirements.txt
```

### Usage Patterns

For quick exploration (moderate directory size):

```bash
python file_cataloger.py /path/to/your/directory
```

For serious data archaeology (massive directory structures):

```bash
python file_cataloger.py /path/to/your/directory \
  --output dataset.csv \
  --batch-size 50000 \
  --workers 8 \
  --memory-limit 6 \
  --verbose
```

### Command Arguments Explained

| Argument         | Short | What It Actually Does                            | Default            |
|------------------|-------|--------------------------------------------------|--------------------|
| `--output`       | `-o`  | Where to save your CSV treasure map              | `file_catalog.csv` |
| `--extensions`   | `-e`  | Focus on specific file types (e.g., `.jpg .png`) | All files          |
| `--batch-size`   | `-b`  | Files per processing batch (lower = less RAM)    | 100000             |
| `--workers`      | `-w`  | CPU cores to utilize                             | CPU count - 1      |
| `--memory-limit` | `-m`  | RAM ceiling in GB                                | 4                  |
| `--verbose`      | `-v`  | Get chatty progress updates                      | False              |

## 🧠 From Raw Data to Pattern Discovery

Once you've generated your dataset, the real exploration begins. Here are some starting points for pattern mining:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (for datasets that fit in memory)
df = pd.read_csv('file_catalog.csv')

# Directory depth analysis
df['directory_depth'] = df['Path'].apply(lambda x: len(x.split('/')))
depth_counts = df['directory_depth'].value_counts().sort_index()

# Filename pattern extraction
df['filename_pattern'] = df['Filename'].str.extract(r'(.+?)[\d_-]')[0]
pattern_counts = df['filename_pattern'].value_counts().head(20)

# Visualization example
plt.figure(figsize=(12, 6))
sns.barplot(x=pattern_counts.index, y=pattern_counts.values)
plt.title('Top 20 Filename Patterns')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('filename_patterns.png')
```

### Scaling to Massive Datasets

For datasets too large for memory, use chunked processing:

```python
def analyze_chunked(csv_path, chunk_size=100000):
    patterns = {}
    directory_depths = {}

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # Update pattern counts
        chunk_patterns = chunk['Filename'].str.extract(r'(.+?)[\d_-]')[0]
        for pattern, count in chunk_patterns.value_counts().items():
            if pd.notna(pattern):
                patterns[pattern] = patterns.get(pattern, 0) + count

        # Update directory depth counts
        chunk['depth'] = chunk['Path'].apply(lambda x: len(x.split('/')))
        for depth, count in chunk['depth'].value_counts().items():
            directory_depths[depth] = directory_depths.get(depth, 0) + count

    return patterns, directory_depths
```

## 💡 Real-World Applications

- **Dataset Organization**: Identify inconsistent naming patterns across research datasets
- **Migration Planning**: Map your current file structure before a system migration
- **Storage Optimization**: Find duplicate organization patterns wasting space
- **Digital Asset Management**: Discover how your digital assets are actually organized vs. how you think they are

## 🛠️ When Things Go Sideways

| Issue                 | The Fix                                                                            |
|-----------------------|------------------------------------------------------------------------------------|
| **Permission Errors** | Run with appropriate privileges or use `--verbose` to identify problem directories |
| **Memory Pressure**   | Decrease `--batch-size` or increase `--memory-limit` if you have RAM to spare      |
| **Performance Crawl** | Bump up `--workers` to match your available CPU cores                              |
| **CSV Too Massive**   | Use the chunked processing approach for analysis (see example code)                |

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
