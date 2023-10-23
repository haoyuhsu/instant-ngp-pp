# Utility function to retrieve 3D models from Objaverse (https://objaverse.allenai.org/objaverse-1.0/)
import objaverse
import random
import multiprocessing

processes = multiprocessing.cpu_count()

def load_lvis_annotations():
    lvis_annotations = objaverse.load_lvis_annotations()
    lvis_annotations = {k.lower(): v for k, v in lvis_annotations.items()}
    return lvis_annotations

def retrieve_object_from_objaverse(obj_name):
    obj_name = obj_name.lower().replace(' ', '_')
    lvis_annot = load_lvis_annotations()
    if obj_name in lvis_annot:
        print('Found object in Objaverse:', obj_name)
        obj_id_list = lvis_annot[obj_name]
        obj_id = random.choice(obj_id_list)
        obj_info = objaverse.load_objects(
            uids=[obj_id],
            download_processes=processes
        )
        return obj_info
    else:
        print('Object not found in Objaverse: ', obj_name, '. Try generate one.')
        return None

if __name__ == '__main__':
    obj_name = 'apple'
    obj_info = retrieve_object_from_objaverse(obj_name)
    print(obj_info)