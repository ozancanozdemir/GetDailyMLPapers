import feedparser
import urllib.parse

# Config
CATEGORIES = ['cs.AI', 'cs.CV', 'cs.CL', 'cs.LG', 'stat.ML', 'q-fin.CP']
MAX_RESULTS = 20
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
        title = entry.title.
