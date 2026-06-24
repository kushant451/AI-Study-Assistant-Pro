from rag.citation_engine import chunks_to_plain_text

STYLE_PROMPTS = {
    "brief": "Summarize the following study material in 3-4 short bullet points covering only key ideas.",
    
    "detailed": """
Create complete university exam notes from the document.

- Cover ALL topics
- Do not skip any section
- Use numbered headings
- 3–5 subpoints per heading
- Each subpoint 2–4 lines
- Include definitions, examples, advantages, disadvantages
- Rewrite in study-note format (not copy-paste)
"""
}


def summarize(client, chunks, style="brief", query=""):
    batch_size = 5
    query = query or "Summarize the document"

    # Split into batches
    chunk_batches = [
        chunks[i:i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]

    batch_summaries = []

    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["brief"])


    for batch in chunk_batches:

        context = chunks_to_plain_text(batch, limit=len(batch))

        # 🔥 SAFE TOKEN CONTROL
        context = context[:1600]

        if style == "brief":
            batch_limit = "Max 50-60 words"
        else:
            batch_limit = "Max 80-100 words with structured points"

        batch_system_prompt = f"""
{system_prompt}

IMPORTANT:
- Summarize ONLY this section
- {batch_limit}
- Keep headings and key concepts
"""

        user_prompt = f"""
User Request:
{query}

Material:
{context}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": batch_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        batch_summaries.append(response.choices[0].message.content)


    combined_summary = "\n".join(batch_summaries)

    # 🔥 FINAL SAFETY LIMIT
    combined_summary = combined_summary[:4000]

    final_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
Create a final structured summary from the following content:

{combined_summary}
"""
            },
        ],
        temperature=0.3,
    )

    return final_response.choices[0].message.content