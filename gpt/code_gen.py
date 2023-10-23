# Use GPT-4 API to generate code for a given prompt
import os
import openai

openai.organization = "org-3RBckaRMgqYfrez6l1XnkmWi"
openai.api_key = os.getenv("OPENAI_API_KEY")

GPT_MODEL = "gpt-4-0613"

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

def generate_response(edit_instruction):
    assert isinstance(edit_instruction, str), "`prompt` should be a string"
    # system message: WHAT IS THE ROLE OF GPT MODEL?
    system_msg = 'You are a helpful assistant with high intelligence who understands how to write codes.'
    # pre-task message to explain details of the task and how the response should be formatted
    with open('gpt/prompts/pre_task_limit_actions.txt', 'r') as file:
        pre_task_prompt = file.read()
    # user message as the query
    query = '''
Scene representation: NeRF
Language description: {}
'''.format(edit_instruction)
    # concatenate system message and user message
    user_msgs = pre_task_prompt + '\n' + query
    response = chat(system_msg, [user_msgs])
    return response

def extract_code_from_response(response):
    # crop out from response, retain only the code part
    output_code = response.split('```python')[1].split('```')[0]
    return output_code

if __name__ == "__main__":
    # test single generate_code
    edit_instruction = "Place an apple on the grass."
    response = generate_response(edit_instruction)
    print(response)

    # # List available GPT models
    # model_lst = openai.Model.list()
    # # print(model_lst.keys())
    # for model in model_lst['data']:
    #     print(model['id']) if model['id'].startswith('gpt') else None

    # with open('prompts/edit_instructions.txt', 'r') as file:
    #     edit_instructions = file.readlines()

    # # output the response as txt file
    # with open('prompts/edit_responses.txt', 'w') as file:
    #     for edit_text in edit_instructions:
    #         if edit_text.startswith('='):
    #             continue
    #         response = generate_response(edit_text)
    #         file.write("Language description: " + edit_text)
    #         file.write(response + '\n')
    #         file.write('=====================================\n')
        