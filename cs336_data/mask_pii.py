import re

def mask_emails(text: str) -> tuple[str, int]:
    # matches standard email patterns (e.g., user.name+tag@domain.com)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    masked_text, count = re.subn(email_pattern, "|||EMAIL_ADDRESS|||", text)
    return masked_text, count

def mask_phone_numbers(text: str) -> tuple[str, int]:
    # Matches common US formats: 123-456-7890, (123) 456-7890, 123.456.7890, +1 123-456-7890
    # Uses negative lookbehinds/lookaheads (?<!\d) and (?!\d) to avoid matching sequences inside larger numbers
    phone_pattern = r'(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)'
    
    masked_text, count = re.subn(phone_pattern, "|||PHONE_NUMBER|||", text)
    return masked_text, count

def mask_ips(text: str) -> tuple[str, int]:
    # Matches 4 groups of numbers ranging from 0 to 255 separated by dots.
    # We build the 0-255 segment regex first for readability:
    # 25[0-5]        : 250 - 255
    # 2[0-4][0-9]    : 200 - 249
    # [01]?[0-9][0-9]? : 0 - 199
    octet = r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
    
    # Combine into the full IPv4 pattern
    ip_pattern = rf'\b(?:{octet}\.){{3}}{octet}\b'
    
    masked_text, count = re.subn(ip_pattern, "|||IP_ADDRESS|||", text)
    return masked_text, count