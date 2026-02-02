from github_llm import chat_completion

print(
    chat_completion(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello in one short sentence."},
        ]
    )
)
