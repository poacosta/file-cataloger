import os
from pathlib import Path
import pandas as pd
import argparse
import logging
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import gc


def setup_logging(log_level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def process_directory(directory, root_path, file_extensions=None, chunk_size=10000):
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
    chunk_size : int
        Number of files to process before yielding results

    Yields:
    -------
    list of tuples
        List of (filename, path) tuples for files
    """
    results = []

    try:
        for path in directory.iterdir():
            if path.is_file() and (file_extensions is None or path.suffix.lower() in file_extensions):
                # Get relative path from root directory
                rel_path = str(path.parent.relative_to(root_path))
                results.append((path.name, rel_path))

                # Yield in chunks to avoid excessive memory usage
                if len(results) >= chunk_size:
                    yield results
                    results = []
    except (PermissionError, OSError) as e:
        logging.warning(f"Skipping {directory}: {str(e)}")

    # Yield any remaining results
    if results:
        yield results


def worker_function(directory, root_path, file_extensions):
    """Worker function for multiprocessing."""
    results = []
    for chunk in process_directory(directory, root_path, file_extensions):
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


def catalog_files(root_dir, output_csv, file_extensions=None,
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
    # Calculate max chunks based on memory limit (rough estimate)
    # Assume each file record takes ~200 bytes in memory
    estimated_bytes_per_record = 200
    max_records_in_memory = int((memory_limit_gb * 1024 * 1024 * 1024) / estimated_bytes_per_record)

    # Adjust batch_size if it would exceed memory limit
    if batch_size > max_records_in_memory:
        adjusted_batch_size = max(10000, max_records_in_memory // 2)  # Ensure minimum batch size
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Batch size ({batch_size}) would exceed memory limit of {memory_limit_gb}GB. "
            f"Adjusting to {adjusted_batch_size}."
        )
        batch_size = adjusted_batch_size
    logger = setup_logging()

    # Convert extensions to lowercase set for faster lookup if provided
    file_extensions_set = None
    if file_extensions:
        file_extensions_set = set(ext.lower() for ext in file_extensions)

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

    for chunk_idx, dir_chunk in enumerate(directory_chunks):
        logger.info(f"Processing directory chunk {chunk_idx + 1}/{len(directory_chunks)}")

        # Create a worker pool for this batch of directories
        with mp.Pool(num_workers) as pool:
            process_func = partial(worker_function, root_path=root_path, file_extensions=file_extensions_set)

            # Process directories in parallel
            for batch_results in tqdm(
                    pool.imap_unordered(process_func, dir_chunk),
                    total=len(dir_chunk),
                    desc="Processing directories"
            ):
                file_counter += len(batch_results)

                # Process results in batches to manage memory usage
                if batch_results:
                    filenames, paths = zip(*batch_results)
                    batch_df = pd.DataFrame({
                        'Filename': filenames,
                        'Path': paths
                    })

                    # Write to CSV
                    mode = 'w' if is_first_batch else 'a'
                    header = is_first_batch
                    batch_df.to_csv(output_csv, mode=mode, header=header, index=False)
                    is_first_batch = False

                    total_files += len(batch_df)

                    # Clear references to free memory
                    del batch_df, filenames, paths
                    gc.collect()

                    # Log progress
                    if file_counter >= batch_size:
                        logger.info(f"Processed {total_files} files so far")
                        file_counter = 0

    logger.info(f"Cataloging complete. Total of {total_files} files found.")
    return total_files


def main():
    """Command-line interface for the file cataloger."""
    parser = argparse.ArgumentParser(description='Catalog all files in a directory structure')
    parser.add_argument('root_dir', help='Root directory to scan for files')
    parser.add_argument('--output', '-o', default='file_catalog.csv',
                        help='Output CSV file path (default: file_catalog.csv)')
    parser.add_argument('--extensions', '-e', nargs='+',
                        help='File extensions to include (e.g., .jpg .png), omit to include all files')
    parser.add_argument('--batch-size', '-b', type=int, default=100000,
                        help='Batch size for processing files (default: 100000)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--memory-limit', '-m', type=float, default=4,
                        help='Memory limit in GB (default: 4)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

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

    catalog_files(
        args.root_dir,
        args.output,
        file_extensions,
        batch_size=args.batch_size,
        num_workers=args.workers,
        memory_limit_gb=args.memory_limit
    )


if __name__ == "__main__":
    main()
