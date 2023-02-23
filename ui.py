import webbrowser
import os
import sys
import shutil
from threading import Timer
from flask import Flask, render_template
import matplotlib
import matplotlib.pyplot as plt


# This fixes the issue where Matplotlib gives a threading error
# Fixed using source:
# https://forum.djangoproject.com/t/matplotlib-in-django-starting-a-matplotlib-gui-outside-of-the-main-thread/14732
matplotlib.use('SVG')

from nuscenes.nuscenes import NuScenes
from minf_part1_functions import *

app = Flask(__name__)
HOST = '127.0.0.1'  # only visible to local machine
PORT = 8080

dataroot = sys.argv[1]
version = sys.argv[2]
nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)


def start_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open('http://' + HOST + ':' + str(PORT) + '/')


@app.route('/')
def index():
    return render_template('index.html', nusc=nusc)


@app.route('/scene/<string:token>')
def scene(token):
    scores = [score for score in generate_scores_for_scene(nusc, token) if score['score'] < 1]

    for score in scores:
        out_path = 'static/temp_renders/' + score['annotation'] + '.jpg'
        try:
            nusc.render_annotation(score['annotation'], out_path=out_path)
        except:
            fig = plt.figure(figsize=(18,9))
            ax = fig.add_subplot()
            ax.text(0.1, 0.1, 'NuScenes API was unable to render annotation', fontsize=25, color='red')
            plt.savefig(out_path)

    return render_template('scene.html', nusc=nusc, token=token, scores=scores)


def main():
    temp_render_dir = 'static/temp_renders'
    if os.path.exists(temp_render_dir):
        shutil.rmtree(temp_render_dir)
    os.makedirs(temp_render_dir)

    Timer(1, start_browser).start()
    app.run(host=HOST, port=PORT, debug=True)


if __name__ == '__main__':
    main()
