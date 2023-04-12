import webbrowser
import os
import sys
import shutil
from threading import Timer
from flask import Flask, render_template, redirect
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns

import constants

# This fixes the issue where Matplotlib gives a threading error
# Fixed using source:
# https://forum.djangoproject.com/t/matplotlib-in-django-starting-a-matplotlib-gui-outside-of-the-main-thread/14732
matplotlib.use('SVG')

from nuscenes.nuscenes import NuScenes
from scoring import *

app = Flask(__name__)
HOST = '127.0.0.1'  # only visible to local machine
PORT = 8080

dataroot = sys.argv[1]
version = sys.argv[2]
nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

aggressive = True


def start_browser():
    """
    Starts the browser only on initial run (aka not on auto restart)

    """
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open('http://' + HOST + ':' + str(PORT) + '/')


@app.route('/')
def index():
    """
    Home page endpoint

    Returns: the html file for the home page

    """
    return render_template('index.html', nusc=nusc, aggressive=aggressive)


@app.route('/trigger_aggressive')
def trigger_aggressive():
    """
    Switches between parameter sets then redirects back to home page

    Returns: home page

    """
    global aggressive
    aggressive = not aggressive
    return redirect('/')



@app.route('/scene/<string:token>')
def scene(token):
    """
    Runs the tools analyses on a specific scene in the dataset

    Args:
        token: scene token

    Returns: results of analyses as webpage

    """

    scores = [score for score in generate_scores_for_scene(nusc, token, aggressive) if score['score'] < 1]

    for score in scores:
        out_path = 'static/temp_renders/' + score['annotation'] + '.jpg'
        try:
            nusc.render_annotation(score['annotation'], out_path=out_path)
        except:
            fig = plt.figure(figsize=(18,9))
            ax = fig.add_subplot()
            ax.text(0.1, 0.1, 'NuScenes API was unable to render annotation', fontsize=25, color='red')
            plt.savefig(out_path)

    return render_template('scene.html', nusc=usc, token=token, scores=scores)

@app.route('/dataset_stats')
def dataset_stats():
    """
    Runs the tools analyses over the entire dataset and then returns results as graphs

    Returns: results page

    """

    scores_raw = []
    for scene in nusc.scene:
        scores_raw += generate_scores_for_scene(nusc, scene['token'], aggressive=aggressive)
    scores = [score['score'] for score in scores_raw]

    fig, ax = plt.subplots()
    ax.hist(scores)
    ax.set_title('Distribution of scores over entire dataset')
    ax.set_xlabel('Score')
    ax.set_ylabel('Number of occurrences')
    fig.savefig('static/temp_renders/dataset.svg')
    fig.savefig('static/temp_renders/dataset.pdf')

    fig, ax = plt.subplots(figsize=(6.4, 3))
    labels = ['Perfect score', 'Non-perfect score']
    values = [len([s for s in scores if s >= 1]), len([s for s in scores if s < 1])]
    ax.pie(values, autopct=lambda x: int(x*len(scores)/100), labels=labels)
    ax.set_title('Proportion of perfect to non-perfect scores')
    fig.subplots_adjust(bottom=0.0)
    fig.savefig('static/temp_renders/dataset_pie.svg')
    fig.savefig('static/temp_renders/dataset_pie.pdf')

    fig, ax = plt.subplots()
    ax.hist([s for s in scores if s < 1])
    ax.set_title('Distribution of non-perfect scores over entire dataset')
    ax.set_xlabel('Score')
    ax.set_ylabel('Number of occurrences')
    fig.savefig('static/temp_renders/dataset_danger.svg')
    fig.savefig('static/temp_renders/dataset_danger.pdf')

    fig, ax = plt.subplots()
    long_same_dir = len([0 for s in scores_raw if s['reason'] == 'Longitudinally too close' and s['same_direction'] == True])
    long_opposite_dir = len([0 for s in scores_raw if s['reason'] == 'Longitudinally too close' and s['same_direction'] == False])
    lat = len([0 for s in scores_raw if s['reason'] == 'Laterally too close'])
    other = len([0 for s in scores_raw if s['reason'] == 'Too close'])
    reasons = ['Longitudinally\ntoo close\n(same\ndirection)',
               'Longitudinally\ntoo close\n(opposite\ndirection)',
               'Laterally\ntoo close', 'Too close']
    values = [long_same_dir, long_opposite_dir, lat, other]
    ax.bar(reasons, values)
    ax.set_title('Reasons given for non-perfect scores')
    ax.set_xlabel('Reason for score')
    ax.set_ylabel('Number of occurrences')
    fig.subplots_adjust(bottom=0.2)
    fig.savefig('static/temp_renders/dataset_reasons.svg')
    fig.savefig('static/temp_renders/dataset_reasons.pdf')

    # Create another graph. A heatmap. It will show the locations of other vehicles in relation to the ego
    # where an unsafe score has been given

    fig, ax = plt.subplots(figsize=(6.4, 3))
    x = [s['lat_distance'] for s in scores_raw if s['score'] < 1]
    y = [s['long_distance'] for s in scores_raw if s['score'] < 1]
    ax.hist2d(y, x, [25,10], range=[[-5,50],[-10,10]])
    l = constants.renault_zoe_dims['length']
    w = constants.renault_zoe_dims['width']
    rect = patches.Rectangle((-l, -w/2), l, w, linewidth=1, edgecolor='r', facecolor='w', label='Ego')
    ax.add_patch(rect)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitudinal distance (m)')
    ax.set_ylabel('Lateral distance (m)')
    ax.set_title('Relative translations of other vehicles in unsafe situations')
    ax.legend()
    fig.savefig('static/temp_renders/dataset_heatmap.svg')
    fig.savefig('static/temp_renders/dataset_heatmap.pdf')

    return render_template('dataset_stats.html')


def main():
    """
    Starts the web server

    """

    temp_render_dir = 'static/temp_renders'
    if os.path.exists(temp_render_dir):
        shutil.rmtree(temp_render_dir)
    os.makedirs(temp_render_dir)

    Timer(1, start_browser).start()
    app.run(host=HOST, port=PORT, debug=True)


if __name__ == '__main__':
    main()
