CLEANER_PROMPT = """
You perform a clean-up work for a PDF scanning project. You will be given
raw character data. Your job is to convert it to a valid JSONL-sequence formatted
like this:

```
{ "type": "title", "content": "INKHEART "}
{ "type": "author", "content": "CORNELIA FUNKE" }
{ "type": "chapterName", "content": "A STRANGER IN THE NIGHT" }
{ "type": "sentence", "content": "The moon shone in the rocking horse's eye, and in the \"mouse's eye\", too, when Tolly fetched it out from
under his pillow to see." }
{ "type": "incompleteSentence", "content": "I heard him say that" }
```

Only output text that could be typed as "title", "author", "chapterName", or "sentence".
You MUST ignore the table of contents, the copyright information and the likes: only the core literature text.
If a sentence ends abruptly at the end of the page, mark it as "incompleteSentence".

You must ONLY output the JSONL text, with no ``` marks or any other content.

One "sentence" object MUST include only one sentence.
- This is GOOD: `{ "type": "sentence", "content": "A cat sat on the mat." }`
- This is UNACCEPTABLE: `{ "type": "sentence", "content": "A cat sat on the mat. What a great cat!" }`

The exception is dialog, which should include a narrator's speech after the quotation mark closes:
- `{ "type": "sentence", "content": "\"You're not here forever," he warned me." }`
"""
