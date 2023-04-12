# minf-part1
University of Edinburgh UG4 Project on Automated Identification of Autonomous Vehicle Risk Scenarios.

## Setup
1. Install NuScenes devkit by running ```$ pip install nuscenes-devkit``` in terminal.
2. Download a portion of the full dataset from https://www.nuscenes.org/nuscenes. Mini portion reccomended. A metadata only potion can be used but will not show any renders. 
3. Extract contents of downloaded file to ```data/sets/nuscenes```. Example file structure for mini portion should be as follows:
```
  / (root directory of project)
  |--data
     |--sets
        |--nuscenes
           |maps 
           |samples (not in metadata only portion)
           |sweeps (not in metadata only portion)
           |v1.0-mini (name will change depending on portion in use)
```
4. Install requirements listed in ```requirements.txt```.

## Run
Run the UI using ```python ui.py data/sets/nuscenes [dataset version]```.

If using the mini dataset, use the command ```python ui.py data/sets/nuscenes v1.0-mini```.

A web browser window should be autonomatically opened with the address http://127.0.0.1:8080. 
