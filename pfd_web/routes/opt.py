import os
import json
import urllib
from flask import Blueprint, render_template, request, url_for, redirect
import cv2
import base64
from pfd_web.utils import video
import time
import datetime
import numpy as np2

opt_bp = Blueprint('opt', __name__)

file_path = os.path.abspath(__file__)  # /home/wzt/PFD/pfd_web/routes/opt.py
current_path = os.path.abspath(os.path.dirname(
    file_path) + os.path.sep + ".")  # /home/wzt/PFD/pfd_web/routes
father_path = os.path.abspath(os.path.dirname(
    current_path) + os.path.sep + ".")  # /home/wzt/PFD/pfd_web
root_path = os.path.abspath(father_path + os.path.sep + "..")  # /home/wzt/PFD

data_path = os.path.join(root_path, 'data')  # /home/wzt/PFD/data
image_path = os.path.join(data_path, 'images')  # /home/wzt/PFD/data/images
opt_path = os.path.join(data_path, 'opts')  # /home/wzt/PFD/data/opts

web_path = os.path.join(root_path, "pfd_web") # /home/wzt/PFD/pfd_web
static_path = os.path.join(web_path, "static") # /home/wzt/PFD/pfd_web/static
json_file = os.path.join(data_path, "scenes.json")


@opt_bp.route('/')
def index():
    return list()


@opt_bp.route('/list')
def list():
    global opt_path
    with open(json_file, "r") as f:
        scenes = json.load(f)
        for val in scenes.values():
            val["cover_image"] = os.path.join("/static", val["cover_image"])
        keys = [k for k in scenes.keys()]

        for scene_id in keys:
            via_json = os.path.join(opt_path, "%s.json" % scene_id)
            if not os.path.exists(via_json):
                del scenes[scene_id]
                continue
        return render_template('opt/list.html', scenes=scenes)


@opt_bp.route('/add')
def add_page():
    return render_template('opt/add.html')

@opt_bp.route('/add', methods=['POST'])
def add_opt():
    scene_id = request.form.get("scene_id")
    via = request.form.get("via_json")
    via = json.loads(via)
    with open(os.path.join(opt_path, "%s.json" % scene_id), "w") as f:
        json.dump(via, f, indent=4, ensure_ascii=False)
        return "OK"

@opt_bp.route('/via/<scene_id>', methods=['GET'])
def via_page(scene_id):
    with open(json_file, "r") as f:
        scenes = json.load(f)
        for val in scenes.values():
            val["cover_image"] = os.path.join("/static", val["cover_image"])
        try:
            with open(os.path.join(opt_path, "%s.json" % scene_id), "r+") as optf:
                via_json = json.load(optf)
        except:
            via_json = dict()
        
        return render_template('opt/via.html', cover_image=scenes[scene_id]["cover_image"], scene_id=scene_id, via_json=str(via_json).replace("\'", "\""))


@opt_bp.route('/del', methods=['DELETE'])
def del_opt():
    global opt_path
    scene_id = request.form.get('scene_id')

    try:
        via_json = os.path.join(opt_path, "%s.json" % scene_id)
        if os.path.exists(via_json):
            os.remove(via_json)
    except:
        pass

    return "OK"