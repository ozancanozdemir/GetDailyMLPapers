import feedparser
import urllib.parse

# ---------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------
CATEGORIES = ['cs.AI', 'cs.CV', 'cs.CL', 'cs.LG', 'stat.ML', 'q-fin.CP']
MAX_RESULTS = 20
TARGET_HEADER = "## 🔥 Latest Papers"
# ---------------------------------------------------------

def update_readme():
    # 1. RSS Sorgusunu Hazırla
    query_parts = [f'cat:{cat}' for cat in CATEGORIES]
    search_query = ' OR '.join(query_parts)
    
    encoded_query = urllib.parse.quote(search_query)
    
    # API URL
    arxiv_url = f'http://export.arxiv.org/api/query?search_query={encoded_query}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_RESULTS}'
    
    print(f"Data is fetching: {arxiv_url}")
    feed = feedparser.parse(arxiv_url)
    
    if not feed.entries:
        print("No update for today.")
        return

    # 2. Markdown İçeriğini Oluştur
    # DÜZELTME 1: Değişken ismini 'papers_content' yaptım ki aşağıda hata vermesin.
    papers_content = "\n" 
    
    for entry in feed.entries:
        date = entry.published[:10]
        title = entry.title.replace('\n', ' ')
        
        # DÜZELTME 2: Satırın sonuna ] parantezi eklendi.
        tags = [t['term'] for t in entry.tags if t['term'].startswith('cs.') or t['term'].startswith('stat.') or t['term'].startswith('q-fin.')]
        
        tags_str = ", ".join(tags)
        
        papers_content += f"- **{title}**\n"
        papers_content += f"  - *Date:* {date} | *Categories:* `{tags_str}`\n"
        papers_content += f"  - [Access the article]({entry.link})\n\n"
    
    # 3. README Dosyasını Güncelle
    try:
        with open('README.md', 'r', encoding='utf-8') as file:
            content = file.read()

        if TARGET_HEADER in content:
            print(f"We insert papers to the appropriate point: {TARGET_HEADER}")

            # Başlıktan itibaren ikiye böl
            parts = content.split(TARGET_HEADER)

            # Footer (imza) kontrolü
            footer_marker = "<div align=\"center\">"
            footer = ""
            # parts[1] başlığın altındaki kısımdır
            if footer_marker in parts[1]:
                footer_index = parts[1].find(footer_marker)
                footer = "\n" + parts[1][footer_index:]
            
            # Yeni içeriği birleştir
            new_content = parts[0] + TARGET_HEADER + "\n" + papers_content + footer
            
            with open('README.md', 'w', encoding='utf-8') as file:
                file.write(new_content)
            print("README is succesfully updated!")
        else:
            print(f"Error: Marker '{TARGET_HEADER}' not found in README.md.")
            
    except FileNotFoundError:
        print("Error: No README.md available!")

if __name__ == "__main__":
    update_readme()
