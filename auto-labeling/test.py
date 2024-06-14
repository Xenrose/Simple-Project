from pathlib import Path
from collections import OrderedDict
import json, sys, os


try:
    from ultralytics import YOLO
except:
    os.system('pip install -r requirements.txt')
    from ultralytics import YOLO



def export_json(bbox_info, output_path):
    output = OrderedDict()
    
    objects = []
    for i in range(len(bbox_info.boxes.cls)):
        objects.append(\
            OrderedDict({"id": i+1,
                         "clsss": bbox_info.names[int(bbox_info.boxes.cls[i])],
                         "confidence": float(bbox_info.boxes.conf[i]),
                         "bbox": bbox_info.boxes.xyxy[i].tolist()}))
        
    output["objects"] = objects
    output["info"] = \
        OrderedDict({"path": bbox_info.path,
                     "width": bbox_info.boxes.orig_shape[0],
                     "height": bbox_info.boxes.orig_shape[1],
                     "class_list": bbox_info.names})


    json.dump(output,
              open(Path(output_path, Path(bbox_info.path).stem + ".json"), 'w', encoding='utf-8'),
              indent=4,
              ensure_ascii=False)
    
    



if __name__ == "__main__":
    code_path, input_path, output_path = map(Path, sys.argv)
    # test_path = [".", "sample_data", "auto_bbox"]
    # code_path, input_path, output_path = map(Path, test_path)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    model = YOLO('model.safetensors', task='detect') # yolo v10n
    results = model(input_path)  

    for result in results:
        export_json(result, output_path)
        result.save(Path(output_path, Path(result.path).name))
