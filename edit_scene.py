from gpt.code_gen import generate_response, extract_code_from_response
from opt import get_opts
from scene_representation import SceneRepresentation
from edit_utils import *

def run_scene_editing(hparams):

    # Load the required scene, dataset and NeRF model
    scene = SceneRepresentation(hparams)

    # Generate the code
    edit_text = hparams.edit_text
    response = generate_response(edit_text)
    print("Editing Instructions:", edit_text)
    print("Response:\n", response)

    # sample response for debugging
#     response = '''
# Explain: 
# The task specifies to place an apple (a 3D object) on the sand (another 3D object) in the 3D scene. We will break this task into two subtasks: retrieve the apple object from the database, locate the sand object in the scene, and then place the apple on it.

# Subtasks:
# (1) Retrieve the 3D object representation of an apple from the database using the Retrieve3DObjectFromDatabase function.
# (2) Localize the 3D object "sand" in the current scene using the Localize3DObjectFromScene function.
# (3) Insert the 3D apple object in the location of the sand in the scene using the Insert3DObjectIntoScene function.

# Code:
# ```python
# def placeAppleOnSand(scene):
#     # Subtask 1: Retrieve 3D object representation of an apple
#     apple_object = Retrieve3DObjectFromDatabase('apple')
    
#     # Subtask 2: localize sand in the scene
#     sand_location = Localize3DObjectFromScene(scene, 'sand')
    
#     # Subtask 3: insert the apple object at the location of the sand
#     new_scene = Insert3DObjectIntoScene(scene, apple_object, sand_location)

#     return new_scene

# # Execute the function with the given scene
# new_scene = placeAppleOnSand(scene)
#     '''

    output_code = extract_code_from_response(response)

    # Execute the code
    exec(output_code)

    # run rendering after all operations
    scene.render_scene(skip_render_NeRF=True)

if __name__ == '__main__':
    hparams = get_opts()
    run_scene_editing(hparams)