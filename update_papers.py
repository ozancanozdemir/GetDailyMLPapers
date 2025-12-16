import feedparser
import urllib.parse

# Config
CATEGORIES = ['cs.AI', 'cs.CV', 'cs.CL', 'cs.LG', 'stat.ML', 'q-fin.CP']
MAX_RESULTS = 5000
TARGET_HEADER = "## 🔥 Latest Papers"
README_FILE = 'README.md'

def update_readme():
    # Read existing README
    try:
        with open(README_FILE, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print("Error: README not found.")
        return

    # Build API query
    query_parts = [f'cat:{cat}' for cat in CATEGORIES]
    search_query = ' OR '.join(query_parts)
    encoded_query = urllib.parse.quote(search_query)
    
    # Fetch data from ArXiv
    url = f'http://export.arxiv.org/api/query?search_query={encoded_query}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_RESULTS}'
    print(f"Fetching: {url}")
    feed = feedparser.parse(url)
    
    if not feed.entries:
        print("No data found.")
        return

    # Process entries
    new_entries_md = ""
    count = 0
    
    for entry in feed.entries:
        link = entry.link
        
        # Check duplicates
        if link in content:
            continue
            
        # Extract metadata
        # HATA BURADAYDI, ŞİMDİ DÜZELTİLDİ:
        title = entry.title.replace('\n', ' ')
        date = entry.published[:10]
        
        # Filter tags
        tags = [t['term'] for t in entry.tags if any(t['term'].startswith(p) for p in ['cs.', 'stat.', 'q-fin.'])]
        tags_str = ", ".join(tags)
        
        # Format Markdown
        new_entries_md += f"- **{title}**\n"
        new_entries_md += f"  - 📅 {date} | 🏷️ `{tags_str}`\n"
        new_entries_md += f"  - [Read Paper]({link})\n\n"
        
        count += 1

    # Update README
    if count > 0:
        if TARGET_HEADER in content:
            # Split content at header
            parts = content.split(TARGET_HEADER)
            
            # Construct new content (Preserve old papers)
            final_content = parts[0] + TARGET_HEADER + "\n\n" + new_entries_md + parts[1]
            
            # Write changes
            with open(README_FILE, 'w', encoding='utf-8') as file:
                file.write(final_content)
            
            print(f"Success: Added {count} new papers.")
        else:
            print(f"Error: Header '{TARGET_HEADER}' not found.")
    else:
        print("No new papers to add.")

if __name__ == "__main__":
    update_readme()
