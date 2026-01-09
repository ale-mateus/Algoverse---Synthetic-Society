from typing import Any

try:
    from openai.types.chat import * 
except Exception:
    pass

class ParsedChatCompletion(Any):
    pass

class ParsedChoice(Any):
    pass
