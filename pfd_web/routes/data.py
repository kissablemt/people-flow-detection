import os
import pandas as pd
import numpy as np
import datetime
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from flask import Blueprint, render_template, request, url_for, redirect
import json

data_bp = Blueprint('data', __name__)

file_path = os.path.abspath(__file__)  # /home/wzt/PFD/pfd_web/routes/data.py
current_path = os.path.abspath(os.path.dirname(
    file_path) + os.path.sep + ".")  # /home/wzt/PFD/pfd_web/routes
father_path = os.path.abspath(os.path.dirname(
    current_path) + os.path.sep + ".")  # /home/wzt/PFD/pfd_web
root_path = os.path.abspath(father_path + os.path.sep + "..")  # /home/wzt/PFD

data_path = os.path.join(root_path, 'data')  # /home/wzt/PFD/data
log_path = os.path.join(data_path, 'logs')  # /home/wzt/PFD/data/logs

scene_json_file = os.path.join(data_path, "scenes.json")

@data_bp.route('/')
def index():
    return list()

@data_bp.route('/test')
def test():
    df = pd.read_csv(os.path.join(log_path, "video4_kp_ct.csv"))
    df.columns = ['frame', 'count', 'total']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["frame"], y=df["count"],
                        mode='lines',
                        name='number'))
    fig.add_trace(go.Scatter(x=df["frame"], y=df["total"],
                        mode='lines',
                        name='total'))
    context = fig.to_json()
    return render_template("/data/test.html", title = 'Home', context = context)

def add(scene_id, dataframe):
    log_dir = os.path.join(log_path, scene_id)
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, "{}.csv".format(str(pd.Timestamp(datetime.datetime.now())).replace(" ", "-")))
    dataframe.to_csv(log_filepath, mode='w', index=False)
    print("log save to %s" % log_filepath)


@data_bp.route('/list')
def list():
    with open(scene_json_file, "r") as f:
        scenes = json.load(f)
        for val in scenes.values():
            val["cover_image"] = os.path.join("/static", val["cover_image"])
        return render_template('data/list.html', scenes=scenes)

@data_bp.route('/list_logs/<scene_id>', methods=['GET'])
def list_logs(scene_id):
    log_dir = os.path.join(log_path, scene_id)
    os.makedirs(log_dir, exist_ok=True)
    logs = os.listdir(log_dir)
    
    with open(scene_json_file, "r") as f:
        scenes = json.load(f)
        if scene_id not in scenes:
            return redirect(url_for("scene.list", scene=scenes))
        return render_template('data/list-logs.html', scene=scenes[scene_id], scene_id=scene_id, logs=logs)
    
@data_bp.route('/show', methods=['GET'])
def show_log():
    log = request.args.get("log")
    scene_id = request.args.get("scene_id")

    log_file = os.path.join(log_path, scene_id, log)
    # log_file = os.path.join(log_path, "video4_kp_ct.csv")
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df.columns = ['time', 'people_count', 'people_total', 'mask_count']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["time"], y=df["people_count"],
                            mode='lines',
                            name='people count'))
        fig.add_trace(go.Scatter(x=df["time"], y=df["people_total"],
                            mode='lines',
                            name='people total'))
        fig.add_trace(go.Scatter(x=df["time"], y=df["mask_count"],
                            mode='lines',
                            name='mask count'))
        context = fig.to_json()
        return render_template("/data/show.html", title = 'Home', context = context)
    else:
        return "FAIL"

@data_bp.route('/del', methods=['DELETE'])
def del_log():
    global opt_path
    scene_id = request.form.get('scene_id')
    log = request.form.get('log')

    try:
        log_file = os.path.join(log_path, scene_id, log)
        if os.path.exists(log_file):
            os.remove(log_file)
    except:
        pass

    return "OK"


    