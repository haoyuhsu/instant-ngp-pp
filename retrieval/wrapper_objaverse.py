# Utility function to retrieve 3D models from Objaverse (https://objaverse.allenai.org/objaverse-1.0/)
import objaverse
import random
import multiprocessing
import urllib.request
import os

processes = multiprocessing.cpu_count()

def load_lvis_annotations():
    lvis_annotations = objaverse.load_lvis_annotations()
    lvis_annotations = {k.lower(): v for k, v in lvis_annotations.items()}
    return lvis_annotations


# def download_object(object_url: str) -> str:
#     """Download the object and return the path."""
#     # uid = uuid.uuid4()
#     uid = object_url.split("/")[-1].split(".")[0]
#     print("Downloading object {} from {}".format(uid, object_url))
#     object_dir = os.path.join("./blender/assets")
#     os.makedirs(object_dir, exist_ok=True)
#     tmp_local_path = os.path.join(object_dir, f"{uid}.glb" + ".tmp")
#     local_path = os.path.join(object_dir, f"{uid}.glb")
#     # wget the file and put it in local_path
#     os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
#     urllib.request.urlretrieve(object_url, tmp_local_path)
#     os.rename(tmp_local_path, local_path)
#     # get the absolute path
#     local_path = os.path.abspath(local_path)
#     return local_path


def retrieve_object_from_objaverse(obj_name):
    obj_name = obj_name.lower().replace(' ', '_')
    lvis_annot = load_lvis_annotations()
    if obj_name in lvis_annot:
        print('Found object in Objaverse:', obj_name)
        obj_id_list = lvis_annot[obj_name]
        obj_id = random.choice(obj_id_list)
        annotations = objaverse.load_annotations(uids=[obj_id])
        object_url = annotations[obj_id]['uri']
        # local_path = download_object(object_url)
        obj_info = objaverse.load_objects(
            uids=[obj_id],
            download_processes=processes
        )
        # obj_info format: {obj_id: obj_path}
        obj_id = list(obj_info.keys())[0]   # get the id of the obj file
        obj_path = obj_info[obj_id]         # get the path of the obj file
        local_path = os.path.join('./blender/assets', obj_path.split('/')[-1])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        os.system('mv {} {}'.format(obj_path, local_path))
        new_obj_info = {}
        new_obj_info['object_name'] = obj_name
        new_obj_info['object_id'] = obj_path.split('/')[-1].split('.')[0]
        new_obj_info['object_path'] = local_path
        return new_obj_info
    else:
        print('Object not found in Objaverse: ', obj_name, '. Try generate one.')
        return None


if __name__ == '__main__':
    obj_name = 'apple'
    obj_info = retrieve_object_from_objaverse(obj_name)
    print(obj_info)