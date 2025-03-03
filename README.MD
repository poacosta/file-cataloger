# 📂 File Cataloger: Path & Metadata Explorer

> Unearth the hidden patterns in your massive directory structures with comprehensive metadata extraction for enhanced
> dataset exploration.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ever stared into the abyss of a filesystem with 500,000+ files and wondered "what's actually in here?" File Cataloger
transforms chaotic directory structures into rich, analyzable datasets that reveal both organizational patterns and
content insights through detailed metadata extraction.

This tool transforms raw filesystem hierarchy into a structured dataset with comprehensive metadata:

1. **Path & Name Extraction**: Capture the basic organizational structure
2. **Metadata Mining**: Extract timestamps, sizes, and file characteristics
3. **Image Intelligence**: Detect and analyze image dimensions, formats, and EXIF data
4. **Pattern Discovery**: Enable rich analysis across both organizational and content dimensions

## 🚀 Key Features

- **Scale-Ready**: Efficiently handles 500k+ files across 1M+ directories
- **Metadata-Rich**: Extracts comprehensive file information including timestamps, sizes, and MIME types
- **Image Expertise**: Automatically detects images and extracts dimensions, format, and EXIF data
- **Memory-Conscious**: Batch processing and garbage collection prevent system slowdowns
- **Parallel Performance**: Multiprocessing maximizes throughput without sacrificing stability
- **Flexible Depth**: Choose between fast basic scans or deep metadata extraction
- **Pattern Filtering**: Exclude files by name patterns with SQL-like wildcards

## 📊 Extracted Metadata

### For All Files

- **Filename & Path**: Complete organizational information
- **File Extension**: Standardized lowercase extension
- **Size**: File size in bytes
- **Timestamps**: Creation, modification, and access times
- **MIME Type**: Content type detection

### Additional Image Metadata

- **Dimensions**: Width and height in pixels
- **Format**: Image format identification (JPEG, PNG, etc.)
- **Color Mode**: RGB, CMYK, grayscale, etc.
- **EXIF Timestamp**: Extracted from image metadata when available
- **Image Validation**: Confirms files are actually valid images regardless of extension

## 🧰 The Technical Bits

### Requirements

- Python 3.9+
- pandas (for data transformation)
- tqdm (for sanity-preserving progress bars)
- Pillow (PIL) for image processing

### Setup

```bash
# Clone the repository
git clone https://github.com/poacosta/file-cataloger.git
cd file-cataloger

# Install dependencies
pip install -r requirements.txt
```

## 💡 Usage Examples

### Basic Usage - Full Metadata Extraction

```bash
python file_cataloger.py /path/to/your/directory
```

### Performance-Focused Scan (Skip Detailed Metadata)

```bash
python file_cataloger.py /path/to/your/directory --basic
```

### Focus on Image Files

```bash
python file_cataloger.py /path/to/your/directory --extensions jpg jpeg png webp gif
```

### Exclude Temporary and System Files

```bash
python file_cataloger.py /path/to/your/directory --exclude-filename "temp%" ".DS_Store" "._%" "Thumbs.db"
```

### Optimize for High-Performance Systems

```bash
python file_cataloger.py /path/to/your/directory \
  --workers 16 \
  --memory-limit 16 \
  --batch-size 200000
```

## Command Arguments Explained

| Argument             | Short | Description                                 | Default            |
|----------------------|-------|---------------------------------------------|--------------------|
| `--output`           | `-o`  | Output CSV file path                        | `file_catalog.csv` |
| `--extensions`       | `-e`  | Specific file extensions to include         | All files          |
| `--exclude-filename` | `-x`  | Exclude files matching wildcard patterns    | None               |
| `--batch-size`       | `-b`  | Files per processing batch                  | 100000             |
| `--workers`          | `-w`  | CPU cores to utilize                        | CPU count - 1      |
| `--memory-limit`     | `-m`  | RAM ceiling in GB                           | 4                  |
| `--verbose`          | `-v`  | Enable detailed logging                     | False              |
| `--metadata`         | `-md` | Extract detailed metadata                   | True               |
| `--basic`            |       | Skip detailed metadata (faster)             | False              |

### Wildcard Pattern Examples

The `--exclude-filename` argument supports SQL-like wildcard patterns:
- `%`: Matches any sequence of characters (including zero)
- `_`: Matches exactly one character

Examples:
- `"temp%"` - Excludes files starting with "temp"
- `"%backup%"` - Excludes files with "backup" anywhere in the name
- `"%.tmp"` - Excludes files ending with .tmp
- `".__%"` - Excludes macOS metadata files
- `"log_20__.txt"` - Excludes log files for years 2000-2099

### Examples

Basic mode (default specs only)

```
python file_cataloger.py  C:\Users\pedro\Documents\Images\SM --output C:\Users\pedro\Documents\Datasets\sm-images.csv
2025-02-26 12:59:47 - INFO - Starting catalog of files in C:\Users\pedro\Documents\Images\SM with 11 workers
2025-02-26 12:59:47 - INFO - Finding all directories...
2025-02-26 12:59:53 - INFO - Found 14927 directories to process
2025-02-26 12:59:53 - INFO - Processing directory chunk 1/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.22it/s]
2025-02-26 13:00:20 - INFO - Processing directory chunk 2/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:28<00:00, 35.25it/s]
2025-02-26 13:00:49 - INFO - Processing directory chunk 3/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.37it/s]
2025-02-26 13:01:16 - INFO - Processing directory chunk 4/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:27<00:00, 36.64it/s]
2025-02-26 13:01:43 - INFO - Processing directory chunk 5/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:32<00:00, 30.89it/s]
2025-02-26 13:02:16 - INFO - Processing directory chunk 6/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:34<00:00, 29.20it/s]
2025-02-26 13:02:50 - INFO - Processing directory chunk 7/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:30<00:00, 33.16it/s]
2025-02-26 13:03:20 - INFO - Processing directory chunk 8/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:32<00:00, 30.33it/s]
2025-02-26 13:03:53 - INFO - Processing directory chunk 9/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:14<00:00, 67.29it/s]
2025-02-26 13:04:08 - INFO - Processing directory chunk 10/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:24<00:00, 41.66it/s]
2025-02-26 13:04:33 - INFO - Processing directory chunk 11/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:25<00:00, 39.54it/s]
2025-02-26 13:04:58 - INFO - Processing directory chunk 12/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.60it/s]
2025-02-26 13:05:25 - INFO - Processing directory chunk 13/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.21it/s]
2025-02-26 13:05:52 - INFO - Processing directory chunk 14/15
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:25<00:00, 38.87it/s]
2025-02-26 13:06:18 - INFO - Processing directory chunk 15/15
Processing directories: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 927/927 [00:24<00:00, 37.20it/s]
2025-02-26 13:06:43 - INFO - Cataloging complete. Total of 97472 files found.
```

Advanced mode with pattern exclusion

```
python file_cataloger.py C:\Users\pedro\Documents\Images\MDT \
  --output C:\Users\pedro\Documents\Datasets\mdt-images.csv \
  --exclude-filename "%temp%" "._%" ".DS_Store" \
  --batch-size 50000 --workers 8 --memory-limit 5 --verbose
2025-02-26 13:10:06 - INFO - Using 3 filename exclusion patterns
2025-02-26 13:10:06 - INFO - Starting catalog of files in C:\Users\pedro\Documents\Images\MDT with 8 workers
2025-02-26 13:10:06 - INFO - Finding all directories...
2025-02-26 13:10:17 - INFO - Found 8341 directories to process
2025-02-26 13:10:17 - INFO - Processing directory chunk 1/9
Processing directories: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:04<00:00, 244.48it/s]
2025-02-26 13:10:22 - INFO - Processing directory chunk 2/9
Processing directories: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:06<00:00, 164.61it/s]
2025-02-26 13:10:28 - INFO - Processing directory chunk 3/9
Processing directories:  97%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌          | 971/1000 [00:19<00:00, 34.15it/s]2025-02-26 13:10:47 - INFO - Processed 50078 files so far
Processing directories: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:20<00:00, 49.81it/s]
[...]
2025-02-26 13:13:04 - INFO - Cataloging complete. Total of 371091 files found.
```

## 📊 Real-World Applications

- **Digital Asset Management**: Analyze image collections by dimensions, format, and creation dates
- **Photography Workflows**: Identify inconsistent image formats or metadata across photo libraries
- **Storage Optimization**: Find duplicate or similar images based on dimensions and other properties
- **Content Organization**: Discover patterns in file organization vs. actual content characteristics
- **Time Series Analysis**: Track file creation patterns across timestamps and directories
- **System Cleanup**: Identify temporary files and exclude them during processing

## 🛠️ Troubleshooting

| Issue                     | Solution                                                   |
|---------------------------|------------------------------------------------------------|
| **Slow Image Processing** | Use `--basic` flag to skip detailed metadata extraction    |
| **Memory Pressure**       | Decrease `--batch-size` or increase `--memory-limit`       |
| **PIL Errors**            | Ensure Pillow is properly installed (`pip install pillow`) |
| **Incomplete EXIF Data**  | Normal - not all images contain complete EXIF information  |
| **Need to Skip Files**    | Use `--exclude-filename` with appropriate patterns         |

## 🔮 Future Enhancements

- **Content Similarity**: Add perceptual hashing for image similarity detection
- **EXIF Deep Dive**: Extract GPS coordinates, camera models, and other rich EXIF data
- **Duplicate Detection**: Add file content hashing (currently stubbed but commented out for performance)
- **Format Validation**: Detect file format mismatches between extension and actual content
- **Interactive Visualization**: Web dashboard for exploring the dataset visually
- **Advanced Filtering**: Expand pattern matching to include path-based exclusions

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.