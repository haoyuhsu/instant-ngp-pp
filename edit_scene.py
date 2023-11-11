from gpt.code_gen import generate_response, extract_code_from_response
from opt import get_opts
from scene_representation import SceneRepresentation
from edit_utils import *
from gpt.LMP import setup_LMP

def run_scene_editing(hparams):

    # Load the required scene, dataset and NeRF model
    scene = SceneRepresentation(hparams)

    # Generate the code and execute
    lmps = setup_LMP()
    edit_lmp = lmps['plan_ui']
    edit_lmp(hparams.edit_text)

    # run rendering after all operations
    scene.render_scene(skip_render_NeRF=True)

if __name__ == '__main__':
    hparams = get_opts()
    run_scene_editing(hparams)