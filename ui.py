import webbrowser
import os
from threading import Timer
from flask import Flask, render_template
import webbrowser

from nuscenes.nuscenes import NuScenes
from minf_part1_functions import *
from my_nuscenes_functions import *

app = Flask(__name__)
HOST = '127.0.0.1'
PORT = 8080

dataroot = 'data/sets/nuscenes'
# dataroot = '/Volumes/kingston/v1.0-mini'
# nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)

def start_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open('http://' + HOST + ':' + str(PORT) + '/')

@app.route('/')
def index():
    return render_template('index.html', nusc=nusc)

@app.route('/scene/<str:token>')
def scene(token):
    return render_template('scene.html', nusc=nusc, token=token)

def main():


    Timer(1, start_browser).start()
    app.run(host=HOST, port=PORT, debug=True)

if __name__ == '__main__':
    main()