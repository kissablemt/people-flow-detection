import os
import json
import urllib
from flask import Blueprint, render_template, request, url_for, redirect
import cv2
import base64
from pfd_web.utils import video
import time
import numpy as np2

scene_bp = Blueprint('scene', __name__)

file_path = os.path.abspath(__file__)  # /home/wzt/PFD/pfd_web/routes/scene.py
current_path = os.path.abspath(os.path.dirname(
    file_path) + os.path.sep + ".")  # /home/wzt/PFD/pfd_web/routes
father_path = os.path.abspath(os.path.dirname(
    current_path) + os.path.sep + ".")  # /home/wzt/PFD/pfd_web
root_path = os.path.abspath(father_path + os.path.sep + "..")  # /home/wzt/PFD

data_path = os.path.join(root_path, 'data')  # /home/wzt/PFD/data
image_path = os.path.join(data_path, 'images')  # /home/wzt/PFD/data/images

web_path = os.path.join(root_path, "pfd_web") # /home/wzt/PFD/pfd_web
static_path = os.path.join(web_path, "static") # /home/wzt/PFD/pfd_web/static
json_file = os.path.join(data_path, "scenes.json")


@scene_bp.route('/')
def index():
    return list()


@scene_bp.route('/list')
def list():
    with open(json_file, "r") as f:
        scenes = json.load(f)
        for val in scenes.values():
            val["cover_image"] = os.path.join("/static", val["cover_image"])
        #     img_path = os.path.join(
        #         data_path, val.get("cover_image", "empty.png"))
        #     if os.path.exists(img_path):
        #         with open(img_path, "rb") as fimg:
        #             val["cover_image"] = "data:image/jpg;base64," + base64.b64encode(
        #                 fimg.read()).decode("utf-8")
        return render_template('scene/list.html', scenes=scenes)

@scene_bp.route('/select')
def select():
    with open(json_file, "r") as f:
        scenes = json.load(f)
        for val in scenes.values():
            val["cover_image"] = os.path.join("/static", val["cover_image"])
        return render_template('scene/select.html', scenes=scenes)


def to_float(s, default=0.0):
    try:
        ret = float(s)
    except:
        ret = default
    return ret


@scene_bp.route('/add', methods=['GET'])
def add_page():
    with open(json_file, "r") as f:
        # with open("／home/chaowei/1.png","rb") as f:
        return render_template('scene/add.html')


@scene_bp.route('/add', methods=['POST'])
def add_scene():
    with open(json_file, "r") as f:
        scenes = json.load(f)

        name = request.form.get("name")
        source = request.form.get("source")
        tag = request.form.get("tag")

        scene_id = time.time()
        create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        update_time = create_time
        cover_image = "images/scene_cover/{}.png".format(scene_id)

        try:
            cap = cv2.VideoCapture(source)
            for i in range(3):
                ret, img = cap.read()
                if not ret:
                    continue
                break
        except Exception as e:
            print("add_scene(): ", e)

        cv2.imwrite(os.path.join(static_path, cover_image), img)
        scenes[scene_id] = {
            "name": name,
            "source": source,
            "cover_image": cover_image,
            "tag": tag,
            
            "predict_score": 0.75,
            "confidence_score": 1.0,
            "max_overlap": 0.75,

            "max_dist": 0.2,
            "min_confidence": 0.9,
            "nms_max_overlap": 0.5,
            "max_iou_distance": 0.9,
            "max_age": 50,
            "n_init": 3,
            "nn_budget": 100,

            "create_time": create_time,
            "update_time": update_time
        }
    with open(json_file, "w") as f:
        json.dump(scenes, f, indent=4, ensure_ascii=False)
        return "OK"

@scene_bp.route('/edit/<scene_id>', methods=['GET'])
def edit_page(scene_id):
    with open(json_file, "r") as f:
        # with open("／home/chaowei/1.png","rb") as f:
        scenes = json.load(f)
        if not scene_id in scenes:
            return redirect(url_for("scene.list", scene=scenes))
        scene = scenes[scene_id]
        return render_template('scene/edit.html', scene=scene, scene_id=scene_id)

@scene_bp.route('/edit', methods=['POST'])
def edit_scene():
    with open(json_file, "r") as f:
        scenes = json.load(f)
        scene_id = request.form.get("scene_id")
        if not scene_id in scenes:
            return redirect(url_for("scene.list", scene=scenes))
    
        print(scenes)
        name = request.form.get("name")
        source = request.form.get("source")
        tag = request.form.get("tag")

        predict_score = request.form.get("predict_score")
        confidence_score = request.form.get("confidence_score")
        max_overlap = request.form.get("max_overlap")

        max_dist = request.form.get("max_dist")
        min_confidence = request.form.get("min_confidence")
        nms_max_overlap = request.form.get("nms_max_overlap")
        max_iou_distance = request.form.get("max_iou_distance")
        max_age = request.form.get("max_age")
        n_init = request.form.get("n_init")
        nn_budget = request.form.get("nn_budget")

        update_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cover_image = "images/scene_cover/{}.png".format(scene_id)

        try:
            cap = cv2.VideoCapture(source)
            for i in range(3):
                ret, img = cap.read()
                if not ret:
                    img = np.zeros((300, 300, 3), dtype=np.uint8)
                    continue
                break
        except Exception as e:
            print("add_scene(): ", e)

        cv2.imwrite(os.path.join(static_path, cover_image), img)
        
        tmp = {
            "name": name,
            "source": source,
            "cover_image": cover_image,
            "tag": tag,

            "predict_score": to_float(predict_score, 0.75),
            "confidence_score": to_float(confidence_score, 1.0),
            "max_overlap": to_float(max_overlap, 0.75),

            "max_dist": to_float(max_dist, 0.2),
            "min_confidence": to_float(min_confidence, 0.9),
            "nms_max_overlap": to_float(nms_max_overlap, 0.5),
            "max_iou_distance": to_float(max_iou_distance, 0.9),
            "max_age": to_float(max_age, 50),
            "n_init": to_float(n_init, 3),
            "nn_budget": to_float(nn_budget, 100),
            "update_time": update_time
        }
        scenes[scene_id].update(tmp)
    with open(json_file, "w") as f:
        json.dump(scenes, f, indent=4, ensure_ascii=False)
        return "OK"

@scene_bp.route('/del', methods=['DELETE'])
def del_scene():
    scene_id = request.form.get('scene_id')
    with open(json_file, "r") as f:
        scenes = json.load(f)
    with open(json_file, "w+") as f:
        try:
            if scene_id in scenes:
                try:
                    cover_image = os.path.join(static_path, scenes[scene_id]["cover_image"])
                    if os.path.exists(cover_image):
                        os.remove(cover_image)
                except:
                    pass
                del scenes[scene_id]
                
                print("delete scene id: ", scene_id)
        except Exception as e:
            print("del_scene():", e)
        json.dump(scenes, f, indent=4, ensure_ascii=False)
        return "OK"



@scene_bp.route('/show/<id>', methods=['GET'])
def show(id):
    with open(json_file, "r") as f:
        scenes = json.load(f)
        if id not in scenes:
            return redirect(url_for("scene.list", scene=scenes))
        return redirect(url_for("video.show", scene=scenes[id], scene_id=id))
