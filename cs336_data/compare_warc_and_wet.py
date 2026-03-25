from fastwarc.warc import ArchiveIterator, WarcRecordType
import sys
import os

from extract_text import extract_text_from_html_bytes

def compare_warc_and_wet():
    target_uri = "http://artoffiction.blogspot.com/2019/04/the-forensic-records-society-by-magnus.html"

    warc_path = "../data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    wet_path = "../data/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"

    html_text = None
    wet_text = None

    try:
        for record in ArchiveIterator(open(warc_path, 'rb'), record_types=WarcRecordType.response):
            if record.headers.get('WARC-Target-URI') == target_uri:
                raw_bytes = record.reader.read()
                html_text = extract_text_from_html_bytes(raw_bytes)
                break
    except FileNotFoundError:
        print(f'error: {warc_path} not found')
        return
    
    try:
        for record in ArchiveIterator(open(wet_path, 'rb'), record_types=WarcRecordType.conversion):
            if record.headers.get('WARC-Target-URI') == target_uri:
                wet_text = record.reader.read().decode('utf-8', errors='replace')
                break
    except FileNotFoundError:
        print(f'error: {wet_path} not found')
        return
    
    print("resiliparse from WARC URI:")
    print(html_text)
    
    print("common crawl from WET:")
    print(wet_text)

if __name__ == "__main__":
    compare_warc_and_wet()