from duckduckgo_search import DDGS


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
        f"Title: {r.get('title','')}\nSnippet: {r.get('body','')}" for r in results
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

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content