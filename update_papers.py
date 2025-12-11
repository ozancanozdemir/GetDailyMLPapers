import feedparser
import datetime

# ArXiv API URL'si (Örnek: Computer Science - AI kategorisi, son 5 makale)
# Başka kategoriler için 'cat:cs.AI' kısmını değiştirebilirsin (örn: cat:cs.CV, cat:stat.ML)
ARXIV_URL = 'http://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=submittedDate&sortOrder=descending&max_results=5'

def update_readme():
    # Veriyi çek
    feed = feedparser.parse(ARXIV_URL)
    
    # Markdown içeriğini oluştur
    markdown_text = "\n"
    for entry in feed.entries:
        date = entry.published[:10]
        markdown_text += f"- **{entry.title}** ({date})\n"
        markdown_text += f"  - [Makaleye Git]({entry.link})\n\n"
    
    # README dosyasını oku
    with open('README.md', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # İşaretleyicilerin (Markers) arasını bul ve değiştir
    start_marker = ''
    end_marker = ''
    
    start_index = content.find(start_marker) + len(start_marker)
    end_index = content.find(end_marker)
    
    if start_index != -1 and end_index != -1:
        new_content = content[:start_index] + markdown_text + content[end_index:]
        
        # Dosyayı kaydet
        with open('README.md', 'w', encoding='utf-8') as file:
            file.write(new_content)
        print("README güncellendi!")
    else:
        print("İşaretleyiciler bulunamadı. Lütfen README.md dosyasına ve ekleyin.")

if __name__ == "__main__":
    update_readme()
