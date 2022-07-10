import yaml
from easydict import EasyDict
def open_cfg(cfg_path):
    
    with open(cfg_path) as f:
        try:
            yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(f)
    return EasyDict(yaml_config)


import json
import numpy as np
def results_to_json(json_file, pred_labels, pred_scores, pred_boxes):
    annos_list = []
    for (pred_label, pred_score, pred_box) in zip(pred_labels, pred_scores, pred_boxes):

        if pred_label == 1:label="Car"
        elif pred_label == 2:label="Pedestrian"
        elif pred_label == 3: label="Cyclist"
        else: label="DontCare"
        
        score = float(pred_score)

        box = pred_box.astype(np.float64)

        annos = {}
        annos['obj_id'] = "1"
        annos['obj_type'] = label
        annos['score'] = score
        annos['psr']={'position':{'x':box[0], 'y':box[1], 'z':box[2]},
                                'scale':{'x':box[3], 'y':box[4], 'z':box[5]}, 
                                'rotation':{'x':0.0, 'y':0.0, 'z':box[6]}} 

        annos_list.append(annos) 

    json.dump(annos_list, open(json_file, "w"), indent=4) 