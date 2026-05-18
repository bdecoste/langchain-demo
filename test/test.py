import urllib.error
import urllib.request
import ssl
import requests

def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL.
    """
#    req = urllib.request.Request(
#        url,
#        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
#    )

    response = requests.get(url, verify=False)
    return response.text

#    context = ssl._create_unverified_context()

#    try:
#        with urllib.request.urlopen(req, timeout = 120) as resp:
#            raw = resp.read()
#    except urllib.error.URLError as e:
#        return f"Fetch failed: {e}"
#    text = raw.decode("utf-8", errors="replace")
#    return text


#URL = "https://www.gutenberg.org/files/64317/64317-0.txt"
URL = "https://raw.githubusercontent.com/bdecoste/text/refs/heads/main/64317-0.txt"

result = fetch_text_from_url(url = URL)
print(result)


