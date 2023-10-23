from gpt.code_gen import generate_response, extract_code_from_response
from opt import get_opts
from scene_representation import SceneRepresentation
from edit_utils import *

def run_scene_editing(hparams):
    # Load the required scene, dataset and NeRF model
    scene = SceneRepresentation(hparams)
    # Generate the code
    edit_instruction = "Place an apple on the sand."
    response = generate_response(edit_instruction)
    print(response)
    output_code = extract_code_from_response(response)
    # Execute the code
    exec(output_code)

if __name__ == '__main__':
    hparams = get_opts()
    run_scene_editing(hparams)