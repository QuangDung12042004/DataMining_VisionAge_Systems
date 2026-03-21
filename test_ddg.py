import time
from duckduckgo_search import DDGS

def test_search():
    print("Testing DDGS for elderly...")
    keywords = ["elderly face", "senior citizen portrait", "old man face"]
    for kw in keywords:
        try:
            print(f"Searching: {kw}")
            with DDGS() as ddgs:
                results = list(ddgs.images(kw, max_results=5))
                print(f"  -> Got {len(results)} results")
                if not results:
                    print("  -> EMPTY!")
        except Exception as e:
            print(f"Exception: {e}")
        time.sleep(2)

if __name__ == "__main__":
    test_search()
