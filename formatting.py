import json
from os.path import join, abspath, dirname, basename, isdir, isfile
import os
from os import listdir
import shutil
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, InMemoryDataset, download_url
import pdb
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_PATH = dirname(__file__)
DATASET_PATH = abspath(join(PROJECT_PATH, 'renders', 'RandomSplitSparseWorld(10)_tidy_train'))
WRITE_PATH = abspath(join(PROJECT_PATH, 'tmp', 'llm_input'))

def list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(list, list_representer)

def process_json(json_data, output_dir, idx):

    
        objects_info = json_data.get('objects', {})
        verbose_constraints = json_data.get('verbose_constraints', [])
        
        # Construct the YAML content
        if idx < 5:
            sample_yaml_content = {
                'Description of the table': 'The table presents a rectangular shape with dimensions measuring 3 units in length and 2 units in width. The coordinate system is centered at the front-left corner of the table, with the origin (0, 0) located at the front-left corner of the tabletop. Consequently, the position of any object specified by the coordinates (x, y) must fall within the range: 0 < x < 3 along the length and 0 < y < 2 along the width. We have the following list of objects: {}. They are arranged neatly on the table.'.format(', '.join(objects_info.keys())),
                'The objects have the following shapes in (l, w)': {},
                'The following object relationships are satisfied': verbose_constraints
            }
            sample_file_path = join(output_dir, 'samples', 'idx={}.yaml'.format(idx))
        else:
            sample_yaml_content = None
              
        # language_query_yaml_content = {
        #         'Task Description': 'The table presents a rectangular shape with dimensions measuring 3 units in length and 2 units in width. The coordinate system is centered at the front-left corner of the table, with the origin (0, 0) located at the front-left corner of the tabletop. Consequently, the position of any object specified by the coordinates (x, y) must fall within the range: 0 < x < 3 along the length and 0 < y < 2 along the width. Your task is to propose an arrangement of the objects so the table is tidy. We have the following list of objects: {}.'.format(', '.join(objects_info.keys())),
        #         'The objects have the following shapes in (l, w)': {},
        #          'To propose the tidy arrangements, strictly follow the following structure': {
        #             '- Object relationships that should be present': '[List the tidy object relationships here]',
        #             '- Locations of the centroids of each object in a python dict': {
        #                 'object_name_1': '[x_1, y_1, 0]',
        #                 'object_name_2': '[x_2, y_2, 0]',
        #             }
        #         }
        # }
        # image_query_yaml_content = {
        #         'Task Description': 'The table presents a rectangular shape with dimensions measuring 3 units in length and 2 units in width. The coordinate system is centered at the front-left corner of the table, with the origin (0, 0) located at the front-left corner of the tabletop. Consequently, the position of any object specified by the coordinates (x, y) must fall within the range: 0 < x < 3 along the length and 0 < y < 2 along the width. Your task is to propose an arrangement of the objects so the table is tidy. We have the following list of objects: {}.'.format(', '.join(objects_info.keys())),
        #         'The objects have the following shapes in (l, w)': {},
        #         'To propose the tidy arrangements, generate an image of the tidy table arrangement with all the object described above.': 'the image should only contain the objects described above.'
        # }

        # language_query_file_path = join(output_dir, 'language_queries', 'idx={}.yaml'.format(idx))
        # image_query_file_path = join(output_dir, 'image_queries', 'idx={}.yaml'.format(idx))
        shapes_info = {}
        centroids_info = {}
        # Populate the YAML content with object details
        for object_name, details in objects_info.items():

            if not object_name.startswith("geometry"):
                extents = details.get('extents', [])
                centroids = details.get('centroid', [])

                # Add object shapes and centroid info if available
                if extents:
                    shapes_info[object_name] = '({:.2f}, {:.2f})'.format(*extents[:2])
                if centroids:
                    centroids_info[object_name] = '({:.2f}, {:.2f}, 0)'.format(*extents[:2])

            
                # Update shapes
        # language_query_yaml_content['The objects have the following shapes in (l, w)'] = shapes_info
        # image_query_yaml_content['The objects have the following shapes in (l, w)'] = shapes_info
        if sample_yaml_content is not None:
                
            sample_yaml_content['The objects have the following shapes in (l, w)'] = shapes_info
            # sample_yaml_content['The locations of the centroids of each object are below'] = centroids_info
            
        # Write the YAML content to a file
        
        # with open(language_query_file_path, 'w') as yaml_file:
        #     yaml.dump(language_query_yaml_content, yaml_file, default_flow_style=False, sort_keys=False)
        
        # with open(image_query_file_path, 'w') as yaml_file:
        #     yaml.dump(image_query_yaml_content, yaml_file, default_flow_style=False, sort_keys=False)

        if sample_yaml_content is not None:
            with open(sample_file_path, 'w') as yaml_file:
                yaml.dump(sample_yaml_content, yaml_file, default_flow_style=False, sort_keys=False)

def process_folders():
    for dir in os.listdir(DATASET_PATH):
        
        if dir == "coffee_table" or dir == "dining_table" or dir == "study_table":
            task_dir = join(DATASET_PATH, dir)
            output_dir = join(WRITE_PATH, dir)
            if not isdir(output_dir):
                os.mkdir(output_dir)
            pdb.set_trace()
            sample_output_dir = join(output_dir, 'samples')
            if not isdir(sample_output_dir):
                os.mkdir(sample_output_dir)
            language_query_output_dir = join(output_dir, 'language_queries')
            if not isdir(language_query_output_dir):
                os.mkdir(language_query_output_dir)
            image_query_output_dir = join(output_dir, 'image_queries')
            if not isdir(image_query_output_dir):
                os.mkdir(image_query_output_dir)
            for i in range(5):
                file = "idx={}.json".format(i)
            
                file_path = os.path.join(task_dir, file)
                # Open and read the JSON file
                with open(file_path, 'r') as json_file:
                        
                    data = json.load(json_file)
                    process_json(data, output_dir, i)
        

process_folders()