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

    """ for the CoRL submission
    python train_ddpm.py -input_mode qualitative -timesteps 1000 -EBM ULA -samples_per_step 10 -step_sizes '2 * self.betas'
    python train_ddpm.py -input_mode qualitative -timesteps 1000 -EBM MALA -samples_per_step 10 -step_sizes '0.0001 * torch.ones_like(self.betas)'
    python train_ddpm.py -input_mode qualitative -timesteps 1000 -EBM HMC -samples_per_step 4 -step_sizes '1 * self.betas ** 1.5'
    """

    # aligned_bottom (ok)
    # aligned_vertical (ok)
    # on_top_of (ok)
    # centered
    # next_to_edge (start here)
    # in
    # all_composed_True
    # all_composed_False
    # symmetry
    # next_to
    # regular_grid

    # model = "StructDiffusion"
    model = "Diffusion-CCSP"
    energy_wrapper = False
    model_relation = "all_composed_None"
    evaluate_relation = model_relation
    # EBM = "ULA" 
    EBM = False

    # pretrained = True

    # if energy_wrapper:
    #     # model_id = "ncy5rau9"
    #     model_id = "w33pzaee"
    #     milestone = 7

    # else: 
    #     # model_id = "hpb9b8rv"
    #     # milestone = 9
    #     # model_id = "7ubfpalx"
    #     # milestone = 10
    #     model_id = "xzvkvr6u"
    #     milestone = 9

    # train_ddpm(
    #     get_args(input_mode='tidy', timesteps=1500, EBM=EBM, energy_wrapper=energy_wrapper, 
    #              samples_per_step=3, wandb_name=f"ctd_EBM_{EBM}_wrapper_{energy_wrapper}_model_relation_{model_relation}", 
    #              model_relation=model_relation, evaluate_relation=evaluate_relation, eval_only=False,
    #              pretrained=pretrained, run_id=model_id),
    #     debug=False, visualize=True, data_only=False, milestone=milestone
    # )

    train_ddpm(
        get_args(model=model, input_mode='tidy', timesteps=1500, EBM=EBM, energy_wrapper=energy_wrapper, 
                 samples_per_step=3, wandb_name=f"EBM_{EBM}_wrapper_{energy_wrapper}_model_relation_{model_relation}", 
                 model_relation=model_relation, evaluate_relation=evaluate_relation, eval_only=False),
        debug=False, visualize=True, data_only=False, 
    )

