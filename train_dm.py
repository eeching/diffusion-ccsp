import jacinle; jacinle.hook_exception_ipdb()
import numpy as np
import torch
import pdb
import sys
import os
from os.path import join, isdir
import argparse

from datasets import GraphDataset
from networks.data_transforms import pre_transform
from train_utils import create_trainer, get_args

if not isdir('data'):
    os.mkdir('data')
if not isdir('logs'):
    os.mkdir('logs')
    
trained_model_dict = {
        "bedroom": {
            "all-composed": [["xgqeq3hk", 29], ["5ku6jtbd", 8]],
            "against-wall": [["fg04t120", 29], ["mkljyvdv", 12]],
            "side-touching": [["a1w6gw23", 28], ["7rodfmhz", 13]],
            "on-left-side": [["bdm7y42q", 28], ["an9cjiqw", 13]],
            "in-front-of": [["vs6sj8xv", 28], ["hmuacovv", 13]],
            "under-window": [["iu4se2h0", 29], ["ab5h6xw1", 13]],
            "at-center": [["6a0p99yh", 29], ["biozw9i4", 29], ["u3g1mqd1", 0]],
            "at-corners": [["669trvrk", 29], ["cxwxkutp", 13]],
        },
        "bookshelf": {
            "all-composed": [["muz8mu58", 29]],
            "wall-contact": [["ss30y5q0", 29]],
            "wall-side": [["zj4n0fy3", 28]],
            "side-of": [["bnsktsf4", 29]],
            "aligned": [["6m8cikpv", 28]],
            "sorted": [["vxj41peq", 21]],
        },
        "dining_table": {
            "all-composed": [["93sspr3v", 8], ["rddcy1z7", 11]],
            "table-edge": [["mhoy4rtv", 17]],
            "table-side": [["l3dn6b4w", 13]],
            "aligned": [["uynrflhx", 16]],
            "side-of": [["alc3diw4", 16]],
            "on-top-of": [["0nrdeowh", 22]],
            "symmetry": [["wxnfwngk", 9]],
            "aligned-in-line": [["4um545uv", 16]],
            "regular-grid": [["u980tizj", 6]],
        }
    }

relation_class_dict = {
    "bedroom": ["all-composed", "against-wall", "side-touching", "on-left-side", "in-front-of", "under-window", "at-center", "at-corners"],
    "bookshelf": ["all-composed", "wall-contact", "wall-side", "side-of", "aligned", "sorted"],
    "dining_table": ["all-composed", "table-edge", "table-side", "aligned", "side-of", "on-top-of", "symmetry", "aligned-in-line", "regular-grid"]
}


def train_ddpm(args, **kwargs):
    if args.pretrained:
        trainer = create_trainer(args, **kwargs)
        trainer.load(kwargs['milestone'], run_id=args.run_id)
    else:
        trainer = create_trainer(args, **kwargs)
    if trainer is None:
        return
    trainer.train()


if __name__ == '__main__':
    """
    to add a new task
    1. run dataset.py to generate the pt files and try evaluation / visualization
    2. change dims in create_trainer() in train_utils.py
    3. change init() and initiate_denoise_fns() in ConstraintDiffuser class of denoise_fn.py
    3. change world.name in Trainer class of ddpm.py
    4. train with debug=True and visualize=True
    5. change wandb project name
    """
    # input_mode = "bedroom"
    # model_relation = "all-composed" 
    # model_relation = "against-wall"
    # model_relation = "side-touching"
    # model_relation = "on-left-side"
    # model_relation = "in-front-of"
    # model_relation = "under-window"
    # model_relation = "at-center"
    # model_relation = "at-corners"

    # input_mode = "bookshelf"
    # model_relation = "all-composed"
    # model_relation = "wall-contact"
    # model_relation = "wall-side"
    # model_relation = "side-of"
    # model_relation = "aligned"
    # model_relation = "sorted"

    # input_mode = "dining_table"
    # model_relation = "all-composed"
    # model_relation = "table-edge"
    # model_relation = "table-side"
    # model_relation = "aligned"
    # model_relation = "side-of"
    # model_relation = "on-top-of"
    # model_relation = "symmetry"
    # model_relation = "aligned-in-line"
    # model_relation = "regular-grid"

    parser = argparse.ArgumentParser(description="Parse input_mode, model_relation, and evaluate_relation")

    parser.add_argument('-i', '--input_mode', type=str, choices=["bedroom", "bookshelf", "dining_table"], required=True, help="The input mode, defaults to 'bedroom'")
    parser.add_argument('-r', '--model_relation', type=str, default="all-composed", help="The model relation, defaults to 'all-composed'")
    parser.add_argument('-model', '--model', type=str, choices=["Diffusion-CCSP", "StructDiffusion"], 
                        default="Diffusion-CCSP", help="The model type, defaults to 'Diffusion-CCSP'")
    parser.add_argument('-ew', '--energy_wrapper', action='store_true', help="Use energy wrapper, defaults to False")
    parser.add_argument('-ebm', '--EBM', type=str, default="", help="Use EBM, defaults to None")
    parser.add_argument('-pt', '--pretrained', action='store_true', help="Use pretrained model, defaults to False")
    
    # Parse the arguments
    args = parser.parse_args()

    model = args.model
    input_mode = args.input_mode
    model_relation = args.model_relation
    energy_wrapper = model_relation
    pretrained = args.pretrained

    if args.EBM == "":
        EBM = False
    else:
        EBM = args.EBM
   
    if pretrained:
        model_id, milestone = trained_model_dict[input_mode][model_relation][-1]
    else:
        model_id, milestone = None, None

    train_ddpm(
        get_args(model=model, input_mode=input_mode, timesteps=1500, EBM=EBM, energy_wrapper=energy_wrapper, 
                 samples_per_step=3, wandb_name=f"EBM_{EBM}_wrapper_{energy_wrapper}_model_relation_{model_relation}", 
                 model_relation=model_relation, evaluate_relation=model_relation, eval_only=False, pretrained=pretrained, run_id=model_id),
        debug=False, visualize=True, data_only=False, milestone=milestone
    )



