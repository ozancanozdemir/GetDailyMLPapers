import feedparser
import urllib.parse

# 'cs.AI' (Artificial Intelligence), 
# 'cs.CV' (Computer Vision), 
# 'cs.LG' (Machine Learning), 
# 'cs.CL' (Computation and Language / NLP),
# 'stat.ML' (Statistics - Machine Learning)
# 'q-fin.CP' (Computational Finance)
CATEGORIES = ['cs.AI', 'cs.CV', 'cs.CL', 'cs.LG','stat.ML','q-fin.CP']

#Maximum article number to be represented
MAX_RESULTS = 20
TARGET_HEADER = "## 🔥 Latest Papers"

# ---------------------------------------------------------

def update_readme():
    
    query_parts = [f'cat:{cat}' for cat in CATEGORIES]
    search_query = ' OR '.join(query_parts)
    
    
    encoded_query = urllib.parse.quote(search_query)
    
   #API
    arxiv_url = f'http://export.arxiv.org/api/query?search_query={encoded_query}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_RESULTS}'
    
    print(f"Data is fetching: {arxiv_url}")
    feed = feedparser.parse(arxiv_url)
    
    if not feed.entries:
        print("No update for today.")
        return

    # Markdown text
    markdown_text = "\n"
    for entry in feed.entries:
        date = entry.published[:10]
        title = entry.title.replace('\n', ' ')
        
        # Find the category
        # Labeled category
        tags = [t['term'] for t in entry.tags if t['term'].startswith('cs.') or t['term'].startswith('stat.') or t['term'].startswith('q-fin.')
        tags_str = ", ".join(tags)
        
        markdown_text += f"- **{title}**\n"
        markdown_text += f"  - *Date:* {date} | *Categories:* `{tags_str}`\n"
        markdown_text += f"  - [Access the article.]({entry.link})\n\n"
    
    # Read and update README
    try:
        with open('README.md', 'r', encoding='utf-8') as file:
            content = file.read()

        if TARGET_HEADER in content:
            print(f"We insert papers to the appropriate point: {TARGET_HEADER}")

            #split the article 
            parts = content.split(TARGET_HEADER)

            footer_marker = "<div align=\"center\">"
            footer = ""
            if footer_marker in parts[1]:
                footer_index = parts[1].find(footer_marker)
                footer = "\n" + parts[1][footer_index:]
            
            new_content = parts[0] + TARGET_HEADER + "\n" + papers_content + footer
        
            
            with open('README.md', 'w', encoding='utf-8') as file:
                file.write(new_content)
            print("README is succesfully updated!")
        else:
            print("Error: No marker found in README.md.")
            
    except FileNotFoundError:
        print("Error: No README.md available!")

if __name__ == "__main__":
    update_readme()

