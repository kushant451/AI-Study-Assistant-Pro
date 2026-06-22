def truncate(text, length=400):
    if len(text) <= length:
        return text
    return text[:length] + "..."


def confidence_color(label):
    return {
        "High": "🟢",
        "Medium": "🟡",
        "Low": "🔴",
    }.get(label, "⚪")