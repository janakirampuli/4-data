import os
import sys
import glob
import concurrent.futures
import pathlib
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

from extract_text import *
from gopher_quality_filters import *
from harmful_content import *
from language_identification import *
from mask_pii import *

def process_single_pos_warc_file(warc_path: str, output_path: str, label: str):
    keep_count = 0
    total_count= 0

    with open(output_path, 'w', encoding='utf-8') as outfile:
        try:
            for record in tqdm(ArchiveIterator(open(warc_path, 'rb'), record_types=WarcRecordType.response)):
                total_count += 1
                try:
                    raw_bytes = record.reader.read()
                    text = extract_text_from_html_bytes(raw_bytes)

                    if not text or not text.strip():
                        continue

                    lang, lang_score = identify_language(text)
                    if lang != 'en' or lang_score < 0.65:
                        continue

                    nsfw_label, nsfw_score = classify_nsfw(text)
                    if nsfw_label == 'nsfw' and nsfw_score > 0.5:
                        continue
                    
                    toxic_label, toxic_score = classify_toxic_speech(text)
                    if toxic_label == 'toxic' and toxic_score > 0.5:
                        continue

                    if not gopher_quality_filter(text):
                        continue

                    text, _ = mask_emails(text)
                    text, _ = mask_phone_numbers(text)
                    text, _ = mask_ips(text)

                    clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()

                    outfile.write(f"__label__{label} {clean_text}")
                    keep_count += 1
                except Exception as e:
                    continue
        except Exception as e:
            print(f"error reading warc file {warc_path}: {e}")

    return output_path, keep_count, total_count


if __name__ == "__main__":
    warc_files = glob.glob("./data/url_chunk_a*_warc.warc.gz")
    
    if not warc_files:
        print("No WARC files found matching data/url_chunk_*_warc*")
        sys.exit(1)
        
    out_file = "data/train_positive.txt"

    warc_files.sort()
    
    num_cpus = os.cpu_count() or 4

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    futures = []

    for warc_filepath in warc_files:
        wet_file_name = str(pathlib.Path(warc_filepath).name)
        chunk_out_path = os.path.join("data", f'{wet_file_name}.txt')

        future = executor.submit(
            process_single_pos_warc_file,
            warc_filepath,
            chunk_out_path,
            "high"
        )

        futures.append(future)

    total_kept = 0
    total_processed = 0

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(warc_files)):
        chunk_out, kept, total = future.result()
        total_kept += kept
        total_processed += total

    with open(out_file, 'w', encoding='utf-8') as final_out:
        for warc_filepath in warc_files:
            wet_file_name = str(pathlib.Path(warc_filepath).name)
            chunk_out_path = os.path.join("data", f"{wet_file_name}.txt")

            if os.path.exists(chunk_out_path):
                with open(chunk_out_path, 'r', encoding='utf-8') as chunk_in:
                    final_out.write(chunk_in.read())
    print(f'final positive training data written to {out_file}')