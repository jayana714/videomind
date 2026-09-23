import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chain as chain_module

load_dotenv()

VERIFICATION_SYSTEM_PROMPT = """You are a fact-checker reviewing an AI-generated answer against the source material it was supposed to be based on.
Read the video transcript excerpts and the answer below. Check whether every claim in the answer is actually supported by the excerpts.
Respond in exactly this format:
Grounded: yes or no
Explanation: one or two sentences. If "no", name the specific claim that isn't supported by the excerpts."""

verification_prompt = ChatPromptTemplate.from_messages([
    ("system", VERIFICATION_SYSTEM_PROMPT),
    ("human", "Video transcript excerpts:\n\n{context}\n\nQuestion: {question}\n\nGenerated answer:\n{answer}"),
])

# A second, separate chain — same LLM, different job: checking the first chain's work
verification_chain = verification_prompt | chain_module.llm | StrOutputParser()


def ask_and_verify(query):
    # Step 8: retrieve + generate (reusing the chain built in chain.py)
    # Accessed as chain_module.X (not imported by name) so this always reflects
    # the current retriever/chain — important after chain_module.refresh() runs.
    # Note: retrieval runs twice (once here, once inside chain.invoke) — harmless
    # and free, it's just a local Chroma lookup, kept this way so the verification
    # step has direct access to the exact context that was used.
    docs = chain_module.retriever.invoke(query)
    context = chain_module.format_docs(docs)
    answer = chain_module.chain.invoke(query)

    print(f"\nAnswer:\n{answer}\n")

    # Step 9: verify the answer doesn't contain claims unsupported by the transcript
    verification = verification_chain.invoke({
        "context": context,
        "question": query,
        "answer": answer,
    })

    print(f"Verification:\n{verification}\n")
    return answer, verification


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your .env file first.")
    else:
        query = input("Ask a question about the video: ")
        ask_and_verify(query)