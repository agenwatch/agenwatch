import os, json, asyncio
from groq import Groq

class UniversalLLM:
    def __init__(self, api_key=None, model="llama-3.1-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.client = Groq(api_key=self.api_key)

    async def generate(self, messages, tools=None):
        # No tools
        if tools is None:
            groq_tools = None

        # ToolRegistry (kernel path)
        elif hasattr(tools, "list_specs"):
            groq_tools = [
                {
                    "type": "function",
                    "function": spec
                }
                for spec in tools.list_specs()
            ]

        # List of tools (SDK / mixed path)
        else:
            groq_tools = []

            for t in tools:
                # FunctionTool
                if hasattr(t, "name"):
                    name = t.name
                    description = getattr(t, "description", None) or t.name
                    parameters = getattr(t, "schema", None)

                # Dict-based tool (SDK)
                elif isinstance(t, dict):
                    name = t.get("name")
                    description = t.get("description", name)
                    parameters = t.get("schema")

                else:
                    continue  # Unknown tool type, skip safely
                
                if not name:
                    continue  # Skip tools without a name
                
                groq_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters or {
                            "type": "object",
                            "properties": {}
                        },
                    }
                })

            groq_tools = groq_tools or None

        # ---- SAFE MESSAGE SERIALIZATION (DO NOT REMOVE) ----
        normalized_messages = []
        for m in messages:
            if isinstance(m, dict):
                normalized_messages.append(m)
            else:
                normalized_messages.append({
                    "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                    "content": m.content,
                })
        
        response = await asyncio.to_thread(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=normalized_messages,
                tools=groq_tools,
                tool_choice="auto" if groq_tools else "none"
            )
        )

        msg = response.choices[0].message
        
        # -------------------------------
        # Instrumentation Metadata
        # -------------------------------
        metadata = {
            "llm_call_id": response.id,
            "model": self.model,
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "response_tokens": getattr(response.usage, "completion_tokens", 0),
            "timestamp": response.created,
        }


        if hasattr(msg, "tool_calls") and msg.tool_calls:
            call = msg.tool_calls[0]
            tool_name = call.function.name
            args = json.loads(call.function.arguments)
            return {
                    "text": f'<function="{tool_name}">\n{json.dumps(args)}\n</function>',
                    "instrumentation": metadata,
                }


        return {
                "text": f"<final>\n{msg.content}\n</final>",
                "instrumentation": metadata,
            }


    

__INTERNAL__ = True
