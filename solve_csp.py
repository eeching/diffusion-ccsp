import os
from os.path import join, dirname, isdir
from datasets import GraphDataset, RENDER_PATH
from networks.data_transforms import pre_transform
from train_utils import load_trainer
import argparse
import pdb
from train_dm import trained_model_dict, relation_class_dict


def check_data_graph(dataset_name):
    """ check data balance """
    dataset_kwargs = dict(input_mode='diffuse_pairwise', pre_transform=pre_transform)
    test_dataset = GraphDataset(dataset_name, **dataset_kwargs)
    for data in test_dataset:
        data = data.to('cuda')
        print(data)
        break


def evaluate_model(run_id, milestone, tries=(5, 0), json_name='eval', save_log=True,
                   run_all=False, render=True, run_only=False, resume_eval=False, render_name_extra=None,
                   return_history=False, n_tasks=None, test_name="", **kwargs):
    
    trainer = load_trainer(run_id, milestone, **kwargs)

    trainer.render_dir = trainer.render_dir.replace('train', f'test_{test_name}({n_tasks})')
    os.makedirs(trainer.render_dir, exist_ok=True)
    
    if render_name_extra is not None:
        trainer.render_dir += f'_{render_name_extra}'
        if not isdir(trainer.render_dir):
            os.mkdir(trainer.render_dir)

    trainer.evaluate(milestone, tries=tries, render=render, save_log=save_log,
                     run_all=run_all, run_only=run_only, resume_eval=resume_eval, return_history=return_history, debug=False)

def eval_cases(input_mode, model_relation, args):
   
    model = args.model
    energy_wrapper = args.energy_wrapper
    if args.EBM == "":
        EBM = False
    else:
        EBM = args.EBM
    extra_denoising_steps = args.extra_denoising_steps
    n_tasks = args.n_tasks
    steps = args.steps
    step_sizes = args.step_sizes

    if input_mode == "bedroom":
        world_name = "RandomBedroomWorld"
    elif input_mode == "bookshelf":
        world_name = "RandomShelfWorld"
    elif input_mode == "dining_table":
        world_name = "RandomTabletopWorld"

    model_id, milestone = trained_model_dict[input_mode][model_relation][-1]

    # --- for testing
    train_task = f"{world_name}({n_tasks})_{input_mode}_test/{model_relation}"

    eval_kwargs = dict(tries=(5, 0), json_name='eval', model=model, save_log=False, visualize=True, test_set=True, return_history=False,
                        run_all=True, model_relation=model_relation, evaluate_relation=model_relation, EBM=EBM, 
                        energy_wrapper=energy_wrapper, samples_per_step=steps, eval_only=True, train_task=train_task,
                        extra_denoising_steps=extra_denoising_steps, step_sizes=step_sizes)
 
    test_tasks = {0: f'{world_name}({n_tasks})_{input_mode}_test/{model_relation}'}
    evaluate_model(model_id, input_mode=input_mode, relation=model_relation, milestone=milestone, test_tasks=test_tasks, n_tasks=n_tasks, test_name=model_relation, **eval_kwargs) # False, both


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Parse input_mode, model_relation, and evaluate_relation")

    parser.add_argument('-i', '--input_mode', type=str, choices=["bedroom", "bookshelf", "dining_table"], required=True, help="The input mode, should be one of ['bedroom', 'bookshelf', 'dining_table']")
    parser.add_argument('-r', '--model_relation', type=str, default="all-composed", help="The model relation, defaults to 'all-composed'")
    parser.add_argument('-e', '--evaluate_relation', type=str, help="The evaluate relation, defaults to model_relation if not provided", default="")
    parser.add_argument('-model', '--model', type=str, choices=["Diffusion-CCSP", "StructDiffusion"], default="Diffusion-CCSP", help="The model type, defaults to 'Diffusion-CCSP'")
    parser.add_argument('-ew', '--energy_wrapper', action='store_true', help="Use energy wrapper, defaults to False")
    parser.add_argument('-ebm', '--EBM', type=str, default="", help="Use EBM, defaults to None")
    parser.add_argument('-ed', '--extra_denoising_steps', action='store_true', help="Use extra denoising steps, defaults to True if the flag is present")
    parser.add_argument('-nt', '--n_tasks', type=int, default=20, help="Number of tasks, defaults to 20")
    parser.add_argument('-s', '--steps', type=int, default=3, help="Number of steps, defaults to 3")
    parser.add_argument('-ss', '--step_sizes', type=str, default="1*self.betas", help="Step sizes, defaults to '1*self.betas'")

    
    # Parse the arguments
    args = parser.parse_args()

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

    eval_cases(args.input_mode, args.model_relation, args)