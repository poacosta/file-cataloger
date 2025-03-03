import argparse
import gc
import hashlib
import imghdr
import logging
import mimetypes
import multiprocessing as mp
import os
import re
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS
from tqdm import tqdm

# Initialize mimetypes
mimetypes.init()


def setup_logging(log_level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def get_file_hash(file_path, algorithm='md5', block_size=65536):
    """
    Calculate the hash of a file.

    Parameters:
    -----------
    file_path : Path
        Path to the file
    algorithm : str
        Hash algorithm to use ('md5', 'sha1', 'sha256')
    block_size : int
        Size of blocks to read

    Returns:
    --------
    str
        Hexadecimal hash digest
    """
    try:
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha1':
            hasher = hashlib.sha1()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        with open(file_path, 'rb') as f:
            buf = f.read(block_size)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(block_size)
        return hasher.hexdigest()
    except Exception as e:
        logging.warning(f"Failed to calculate hash for {file_path}: {str(e)}")
        return None


def extract_image_metadata(file_path):
    """
    Extract metadata from an image file.

    Parameters:
    -----------
    file_path : Path
        Path to the image file

    Returns:
    --------
    dict
        Dictionary containing image metadata
    """
    metadata = {
        'width': None,
        'height': None,
        'image_format': None,
        'exif_timestamp': None,
        'color_mode': None
    }

    try:
        with Image.open(file_path) as img:
            metadata['width'] = img.width
            metadata['height'] = img.height
            metadata['image_format'] = img.format
            metadata['color_mode'] = img.mode

            # Extract EXIF data if available
            if hasattr(img, '_getexif') and img._getexif():
                exif = {TAGS.get(tag, tag): value for tag, value in img._getexif().items()}

                # Get timestamp from EXIF data
                for tag in ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']:
                    if tag in exif:
                        metadata['exif_timestamp'] = exif[tag]
                        break
    except UnidentifiedImageError:
        pass  # Not a valid image file
    except Exception as e:
        logging.warning(f"Failed to extract image metadata for {file_path}: {str(e)}")

    return metadata


def get_file_metadata(file_path):
    """
    Get metadata for a file.

    Parameters:
    -----------
    file_path : Path
        Path to the file

    Returns:
    --------
    dict
        Dictionary containing file metadata
    """
    metadata = {
        'size_bytes': None,
        'creation_time': None,
        'modification_time': None,
        'access_time': None,
        'mime_type': None,
        'is_image': False,
        'width': None,
        'height': None,
        'image_format': None,
        'exif_timestamp': None,
        'color_mode': None,
        'file_hash': None
    }

    try:
        # Basic file stats
        stat = file_path.stat()
        metadata['size_bytes'] = stat.st_size
        metadata['creation_time'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
        metadata['modification_time'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        metadata['access_time'] = datetime.fromtimestamp(stat.st_atime).isoformat()

        # MIME type
        mime_type, encoding = mimetypes.guess_type(file_path)
        metadata['mime_type'] = mime_type

        # Check if it's an image
        is_image = mime_type and mime_type.startswith('image/')
        if not is_image:
            is_image = imghdr.what(file_path) is not None

        metadata['is_image'] = is_image

        # Extract image-specific metadata if it's an image
        if is_image:
            image_metadata = extract_image_metadata(file_path)
            metadata.update(image_metadata)

        # Calculate file hash (expensive, so only do it if requested)
        # metadata['file_hash'] = get_file_hash(file_path)

    except Exception as e:
        logging.warning(f"Failed to get metadata for {file_path}: {str(e)}")

    return metadata


def process_file(file_path, root_path):
    """
    Process a single file and extract metadata.

    Parameters:
    -----------
    file_path : Path
        Path to the file
    root_path : Path
        Root directory for relative path calculation

    Returns:
    --------
    dict
        Dictionary containing file information and metadata
    """
    try:
        # Get basic file information
        rel_path = str(file_path.parent.relative_to(root_path))
        file_info = {
            'filename': file_path.name,
            'path': rel_path,
            'extension': file_path.suffix.lower(),
        }

        # Extract metadata
        metadata = get_file_metadata(file_path)
        file_info.update(metadata)

        return file_info
    except Exception as e:
        logging.warning(f"Failed to process {file_path}: {str(e)}")
        return None


def compile_filename_exclusion_patterns(exclude_patterns):
    """
    Compile SQL-like wildcard patterns into regex patterns.

    Parameters:
    -----------
    exclude_patterns : list
        List of SQL-like patterns (e.g., '%word', 'word%', '%word%')

    Returns:
    --------
    list
        List of compiled regex patterns
    """
    if not exclude_patterns:
        return []

    compiled_patterns = []
    for pattern in exclude_patterns:
        # Convert SQL LIKE wildcards to regex
        # % becomes .* (match any character 0 or more times)
        # _ becomes . (match any single character)
        regex_pattern = pattern.replace('%', '.*').replace('_', '.')
        # Anchor the pattern based on wildcards
        if not pattern.startswith('%'):
            regex_pattern = '^' + regex_pattern
        if not pattern.endswith('%'):
            regex_pattern = regex_pattern + '$'

        compiled_patterns.append(re.compile(regex_pattern))

    return compiled_patterns


def should_exclude_file(filename, exclude_patterns):
    """
    Check if a filename matches any of the exclusion patterns.

    Parameters:
    -----------
    filename : str
        Filename to check
    exclude_patterns : list
        List of compiled regex patterns

    Returns:
    --------
    bool
        True if the file should be excluded, False otherwise
    """
    if not exclude_patterns:
        return False

    for pattern in exclude_patterns:
        if pattern.search(filename):
            return True

    return False


def process_directory(directory, root_path, file_extensions=None, exclude_patterns=None, include_metadata=True,
                      chunk_size=10000):
    """
    Process a single directory for files.

    Parameters:
    -----------
    directory : Path
        Directory to process
    root_path : Path
        Root directory for relative path calculation
    file_extensions : set or None
        Set of file extensions to filter by, or None to include all files
    exclude_patterns : list
        List of compiled regex patterns for filename exclusion
    include_metadata : bool
        Whether to include detailed metadata
    chunk_size : int
        Number of files to process before yielding results

    Yields:
    -------
    list of dict
        List of dictionaries containing file information and metadata
    """
    results = []

    try:
        for path in directory.iterdir():
            if path.is_file() and (file_extensions is None or path.suffix.lower() in file_extensions):
                if exclude_patterns and should_exclude_file(path.name, exclude_patterns):
                    continue

                try:
                    if include_metadata:
                        file_info = process_file(path, root_path)
                    else:
                        # Basic file info without detailed metadata
                        rel_path = str(path.parent.relative_to(root_path))
                        file_info = {
                            'filename': path.name,
                            'path': rel_path,
                            'extension': path.suffix.lower(),
                            'size_bytes': path.stat().st_size,
                            'modification_time': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                        }

                    if file_info:
                        results.append(file_info)

                    # Yield in chunks to avoid excessive memory usage
                    if len(results) >= chunk_size:
                        yield results
                        results = []
                except Exception as e:
                    logging.warning(f"Error processing file {path}: {str(e)}")
    except (PermissionError, OSError) as e:
        logging.warning(f"Skipping {directory}: {str(e)}")

    # Yield any remaining results
    if results:
        yield results


def worker_function(directory, root_path, file_extensions, exclude_patterns, include_metadata):
    """Worker function for multiprocessing."""
    results = []
    for chunk in process_directory(directory, root_path, file_extensions, exclude_patterns, include_metadata):
        results.extend(chunk)
    return results


def find_all_directories(root_path):
    """Find all directories under the root path."""
    directories = []
    try:
        for entry in os.scandir(root_path):
            if entry.is_dir():
                dir_path = Path(entry.path)
                directories.append(dir_path)
                directories.extend(find_all_directories(dir_path))
    except (PermissionError, OSError) as e:
        logging.warning(f"Skipping {root_path}: {str(e)}")
    return directories


def catalog_files(root_dir, output_csv, file_extensions=None, exclude_patterns=None, include_metadata=True,
                  batch_size=100000, num_workers=None, memory_limit_gb=4):
    """
    Catalogs all files in a directory structure and exports to CSV.
    Uses multiprocessing and batched processing to handle large directories.

    Parameters:
    -----------
    root_dir : str
        Path to the root directory to scan for files
    output_csv : str
        Path where the CSV output will be saved
    file_extensions : list, optional
        List of file extensions to include. If None, includes all files
    exclude_patterns : list, optional
        List of SQL-like wildcard patterns for filename exclusion
    include_metadata : bool
        Whether to include detailed metadata
    batch_size : int
        Number of files to process in a single batch
    num_workers : int, optional
        Number of worker processes to use. If None, uses CPU count - 1
    memory_limit_gb : float
        Approximate memory limit in GB to prevent OOM errors

    Returns:
    --------
    int
        Total number of files processed
    """
    logger = setup_logging()

    # Calculate max chunks based on memory limit (rough estimate)
    # Assume each file record takes ~200 bytes in memory
    estimated_bytes_per_record = 200
    max_records_in_memory = int((memory_limit_gb * 1024 * 1024 * 1024) / estimated_bytes_per_record)

    # Adjust batch_size if it would exceed memory limit
    if batch_size > max_records_in_memory:
        adjusted_batch_size = max(10000, max_records_in_memory // 2)  # Ensure minimum batch size
        logger.warning(
            f"Batch size ({batch_size}) would exceed memory limit of {memory_limit_gb}GB. "
            f"Adjusting to {adjusted_batch_size}."
        )
        batch_size = adjusted_batch_size

    # Convert extensions to lowercase set for faster lookup if provided
    file_extensions_set = None
    if file_extensions:
        file_extensions_set = set(ext.lower() for ext in file_extensions)

    # Compile exclusion patterns
    compiled_exclude_patterns = compile_filename_exclusion_patterns(exclude_patterns)
    if compiled_exclude_patterns:
        logger.info(f"Using {len(compiled_exclude_patterns)} filename exclusion patterns")

    root_path = Path(root_dir)

    # Set up multiprocessing
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    logger.info(f"Starting catalog of files in {root_dir} with {num_workers} workers")

    # Get all directories first
    logger.info("Finding all directories...")
    all_directories = [root_path] + find_all_directories(root_path)
    logger.info(f"Found {len(all_directories)} directories to process")

    # Process in batches to avoid memory issues
    total_files = 0
    file_counter = 0
    is_first_batch = True

    # Process directories in chunks sized based on memory limit
    # Each directory might use ~500 bytes of memory overhead
    dir_chunk_size = min(1000, int(max_records_in_memory / 50))
    directory_chunks = [all_directories[i:i + dir_chunk_size] for i in range(0, len(all_directories), dir_chunk_size)]

    # Define the columns based on metadata inclusion
    if include_metadata:
        columns = [
            'filename', 'path', 'extension', 'size_bytes',
            'creation_time', 'modification_time', 'access_time',
            'mime_type', 'is_image', 'width', 'height',
            'image_format', 'exif_timestamp', 'color_mode'
        ]
    else:
        columns = ['filename', 'path', 'extension', 'size_bytes', 'modification_time']

    for chunk_idx, dir_chunk in enumerate(directory_chunks):
        logger.info(f"Processing directory chunk {chunk_idx + 1}/{len(directory_chunks)}")

        # Create a worker pool for this batch of directories
        with mp.Pool(num_workers) as pool:
            process_func = partial(
                worker_function,
                root_path=root_path,
                file_extensions=file_extensions_set,
                exclude_patterns=compiled_exclude_patterns,
                include_metadata=include_metadata
            )

            # Process directories in parallel
            for batch_results in tqdm(
                    pool.imap_unordered(process_func, dir_chunk),
                    total=len(dir_chunk),
                    desc="Processing directories"
            ):
                file_counter += len(batch_results)

                # Process results in batches to manage memory usage
                if batch_results:
                    batch_df = pd.DataFrame(batch_results)

                    # Ensure consistent column order
                    for col in columns:
                        if col not in batch_df.columns:
                            batch_df[col] = None
                    batch_df = batch_df[columns]

                    # Write to CSV with proper escaping
                    mode = 'w' if is_first_batch else 'a'
                    header = is_first_batch
                    batch_df.to_csv(
                        output_csv,
                        mode=mode,
                        header=header,
                        index=False,
                        escapechar='\\',  # Use backslash as escape character
                        doublequote=True,  # Double up quotes for escaping
                        quoting=1  # csv.QUOTE_ALL - quote all fields
                    )
                    is_first_batch = False

                    total_files += len(batch_df)

                    # Clear references to free memory
                    del batch_df
                    gc.collect()

                    # Log progress
                    if file_counter >= batch_size:
                        logger.info(f"Processed {total_files} files so far")
                        file_counter = 0

    logger.info(f"Cataloging complete. Total of {total_files} files found.")
    return total_files


def main():
    """Command-line interface for the file cataloger."""
    parser = argparse.ArgumentParser(description='Catalog all files in a directory structure with metadata')
    parser.add_argument('root_dir', help='Root directory to scan for files')
    parser.add_argument('--output', '-o', default='file_catalog.csv',
                        help='Output CSV file path (default: file_catalog.csv)')
    parser.add_argument('--extensions', '-e', nargs='+',
                        help='File extensions to include (e.g., .jpg .png), omit to include all files')
    parser.add_argument('--exclude-filename', '-x', nargs='+',
                        help='Exclude filenames matching these patterns (e.g., "temp%", "%backup%", ".__%"). '
                             'Supports SQL LIKE wildcards: % (any chars), _ (single char)')
    parser.add_argument('--batch-size', '-b', type=int, default=100000,
                        help='Batch size for processing files (default: 100000)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--memory-limit', '-m', type=float, default=4,
                        help='Memory limit in GB (default: 4)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--metadata', '-md', action='store_true', default=True,
                        help='Include detailed metadata (default: True)')
    parser.add_argument('--basic', action='store_false', dest='metadata',
                        help='Only include basic file information (faster)')

    args = parser.parse_args()

    # Set up logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Convert extensions to proper format if provided
    file_extensions = None
    if args.extensions:
        file_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in args.extensions]

    # Get exclude patterns
    exclude_patterns = args.exclude_filename

    start_time = time.time()

    catalog_files(
        args.root_dir,
        args.output,
        file_extensions,
        exclude_patterns=exclude_patterns,
        include_metadata=args.metadata,
        batch_size=args.batch_size,
        num_workers=args.workers,
        memory_limit_gb=args.memory_limit
    )

    elapsed_time = time.time() - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
