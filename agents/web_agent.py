import time
from duckduckgo_search import DDGS

MODEL = "gemini-2.0-flash"


def search_web(query, max_results=5):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        results = []
    return results


def answer_with_web_search(client, query, history_text=""):
    results = search_web(query)

    if not results:
        return "I couldn't find any web results for that query."

    snippets = "\n\n".join(
        f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}"
        for r in results
    )

    system_prompt = (
        "You are a helpful assistant. Use the web search results below to "
        "answer the user's question in your own words. Mention that the "
        "information comes from a web search."
    )

    user_prompt = (
        f"Recent conversation:\n{history_text}\n\n"
        f"Web search results:\n{snippets}\n\nQuestion: {query}"
    )

    prompt = f"{system_prompt}\n\n{user_prompt}"

    # ✅ Fixed: was using OpenAI-style client.chat.completions.create()
    # Now using google-genai SDK
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            wait = min(5 * (attempt + 1), 30)
            print(f"[GEMINI ERROR] attempt {attempt+1}: {type(e).__name__}: {e}")
            time.sleep(wait)

    raise Exception("Gemini API failed after all retries.")