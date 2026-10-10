"""Join translated chunks using source paragraph boundaries."""


def assemble(sources: list[str], translations: list[str]) -> str:
    if len(sources) != len(translations):
        raise ValueError("Source and translation chunk counts differ")
    parts = []
    for index, translation in enumerate(translations):
        if index:
            before, after = sources[index - 1], sources[index]
            whitespace = before[len(before.rstrip()) :] + after[: len(after) - len(after.lstrip())]
            newlines = whitespace.count("\n")
            parts.append("\n\n" if newlines >= 2 else "\n" if newlines else " ")
        parts.append(translation.strip())
    return "".join(parts)
