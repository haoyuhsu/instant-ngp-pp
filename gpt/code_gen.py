# Use GPT-4 API to generate code for a given prompt
import os
import openai

openai.organization = "org-3RBckaRMgqYfrez6l1XnkmWi"
openai.api_key = os.getenv("OPENAI_API_KEY")

GPT_MODEL = "gpt-3.5-turbo"

# Wrapper function for GPT API
def chat(system, user_assistant):
    assert isinstance(system, str), "`system` should be a string"
    assert isinstance(user_assistant, list), "`user_assistant` should be a list"
    system_msg = [{"role": "system", "content": system}]
    user_assistant_msgs = [
        {"role": "assistant", "content": user_assistant[i]} if i % 2 else {"role": "user", "content": user_assistant[i]}
        for i in range(len(user_assistant))]

    msgs = system_msg + user_assistant_msgs
    response = openai.ChatCompletion.create(model=GPT_MODEL,
                                            messages=msgs)
    status_code = response["choices"][0]["finish_reason"]
    assert status_code == "stop", f"The status code was {status_code}."
    return response["choices"][0]["message"]["content"]

# system message: WHAT IS THE ROLE OF GPT MODEL?
system_msg = 'You are a helpful assistant with high intelligence who understands how to write codes.'

# pre-task message to explain details of the task and how the response should be formatted
with open('prompts/pre_task.txt', 'r') as file:
    pre_task_prompt = file.read()

# user message as the query
query_1 = '''
    Scene representation: NeRF
    Language description: Place an apple on the grass.
'''

# concatenate system message and user message
user_msgs = pre_task_prompt + '\n' + query_1

response_1 = chat(system_msg, [user_msgs])

print(response_1)
