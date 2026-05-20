from google import genai


class GenerateGeminiResponse:
    """Handles LLM initialization and response generation using Groq."""

    def __init__(self, api_key="", model="gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self._initialize_llm()

    def _initialize_llm(self):
        if not self.api_key:
            raise ValueError("API key not provided")
        if not self.model:
            raise ValueError("model not provided")
        try:
            self.llm = genai.Client(api_key=self.api_key)

            # self.llm = ChatGroq(
            #     api_key=self.api_key, model=self.model, temperature=0.1, max_tokens=1024
            # )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

        print(f"✓ Initialized Gemini LLM with model: {self.model}")

    def generate_answer(self, prompt, detail=False, **details):
        if not prompt:
            raise ValueError("Prompt is empty")

        print("💭 Generating answer...\n")
        try:
            response = self.llm.models.generate_content(
                model=self.model,
                contents=prompt,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {str(e)}")
        if not detail:
            return response.content

        output = {
            "answer": response.content,
            "sources": details.get("sources", ""),
            "confidence": details.get("confidence", ""),
            "context": details.get("context", ""),
        }

        return output


def main():
    pass


if __name__ == "__main__":
    main()
