# Use GPT-4 API to generate code for a given prompt
import os
import openai
from LMP import LMP

openai.organization = "org-3RBckaRMgqYfrez6l1XnkmWi"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Available GPT models: gpt-4-1106-preview, gpt-4-0613, gpt-4-0314, gpt-4

lmp_planner_cfg = {
    'prompt_fname': 'planner_prompt',
    'model': 'gpt-4-1106-preview',
    'max_tokens': 512,
    'temperature': 0,
    'query_prefix': '# Query: ',
    'query_suffix': '.',
    'stop': [
        '# Query: ',
    ],
    'maintain_session': False,
    'include_context': False,
    'has_return': False,
    'return_val_name': 'ret_val',
    'load_cache': True
}

def setup_LMP():
    # Create LMP
    task_planner = LMP('planner', lmp_planner_cfg, {}, {}, debug=False, env='')

    lmps = {
        'plan_ui': task_planner,
    }
    return lmps

if __name__ == "__main__":
    # test single generate_code
    edit_instruction = "Place an apple on the grass"
    lmps = setup_LMP()
    edit_lmp = lmps['plan_ui']
    generate_LMP(edit_lmp, edit_instruction)

    # # output the response as txt file
    # with open('prompts/edit_responses.txt', 'w') as file:
    #     for edit_text in edit_instructions:
    #         if edit_text.startswith('='):
    #             continue
    #         response = generate_response(edit_text)
    #         file.write("Language description: " + edit_text)
    #         file.write(response + '\n')
    #         file.write('=====================================\n')