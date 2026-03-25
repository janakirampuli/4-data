import glob
import os
import sys
import random
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

from extract_text import *
from gopher_quality_filters import *
from harmful_content import *
from language_identification import *
from mask_pii import *

def process_neg_warc_to_text(warc_path: str, output_path: str, label: str, sample_prob: float = 0.2):
    keep_count = 0
    total_count= 0

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for record in tqdm(ArchiveIterator(open(warc_path, 'rb'), record_types=WarcRecordType.response)):
            total_count += 1
            
            if random.random() > sample_prob:
                continue
                
            try:
                raw_bytes = record.reader.read()
                text = extract_text_from_html_bytes(raw_bytes)
                
                if not text or not text.strip():
                    continue
                    
                lang, lang_score = identify_language(text)
                if lang != 'en' or lang_score < 0.65:
                    continue

                text, _ = mask_emails(text)
                text, _ = mask_phone_numbers(text)
                text, _ = mask_ips(text)
                
                clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
                
                outfile.write(f"__label__{label} {clean_text}\n")
                keep_count += 1
                
            except Exception as e:
                continue

    print(f'finished processing, kep {keep_count} out of {total_count}')
    print(f"data written to {output_path}")

if __name__ == "__main__":
    warc_file = "./data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz" 
    
    if not os.path.exists(warc_file):
        print(f"error: file not found {warc_file}")
        sys.exit(1)
        
    out_file = "data/train_negative.txt"
    
    process_neg_warc_to_text(warc_file, out_file, label="low", sample_prob=0.2)
