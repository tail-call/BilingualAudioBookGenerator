from pypdf import PdfReader
import litellm
from functools import reduce
from datetime import datetime


reader = PdfReader("inkheart_RuLit_Me_789272.pdf")

SYSTEM_PROMPT = """
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
def process_book():
    with open("out.jsonl", "w") as out:
        page_number = 0
        batch_size = 5
        while True:
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time} Working with pages {page_number}..<{page_number + batch_size}")
            response = litellm.completion(
                model="openai/unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M",
                api_base="http://localhost:8080/v1",
                api_key="not-needed",
                messages=[{
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }, {
                    "role": "user",
                    "content": reduce(
                        lambda a, b: a + b,
                        (page.extract_text() for page in reader.pages[
                            page_number:page_number+batch_size
                        ])
                    )
                }]
            )
            out.write(response.choices[0].message.content)
            page_number += batch_size
        print(f"Done writing.")

if __name__ == "__main__":
    process_book()

# OmniVoice
