import os
from os.path import join, dirname, isdir
from datasets import GraphDataset, RENDER_PATH
from networks.data_transforms import pre_transform
from train_utils import load_trainer
import argparse
import pdb


def check_data_graph(dataset_name):
    """ check data balance """
    dataset_kwargs = dict(input_mode='diffuse_pairwise', pre_transform=pre_transform)
    test_dataset = GraphDataset(dataset_name, **dataset_kwargs)
    for data in test_dataset:
        data = data.to('cuda')
        print(data)
        break


def evaluate_model(run_id, milestone, tries=(10, 0), json_name='eval', save_log=True,
                   run_all=False, render=True, run_only=False, resume_eval=False, render_name_extra=None,
                   return_history=False, n_tasks=None, test_name=None, **kwargs):
    
    trainer = load_trainer(run_id, milestone, **kwargs)

    trainer.render_dir = trainer.render_dir.replace('train', f'test_{test_name}({n_tasks})')
    os.makedirs(trainer.render_dir, exist_ok=True)
    
    if render_name_extra is not None:
        trainer.render_dir += f'_{render_name_extra}'
        if not isdir(trainer.render_dir):
            os.mkdir(trainer.render_dir)

    trainer.evaluate(milestone, tries=tries, render=render, save_log=save_log,
                     run_all=run_all, run_only=run_only, resume_eval=resume_eval, return_history=return_history, debug=False)


MILESTONES = {
    'gtqtyd8n': 20, 'dyz4eido': 23,
    'qsd3ju74': 7, 'r26wkb13': 16,
    'ti639t00': 3, 'bo02mwbw': 11,
    '0hhtsh6a': 5, 'qcjqkout': 14,
}


def get_tests(input_mode, test_tiny=False, find_failure_cases=False):
    n = 10 if test_tiny else 100
    if input_mode == 'diffuse_pairwise':
        tests = {i: f"TriangularRandomSplitWorld[64]_({n})_diffuse_pairwise_test_{i}_split" for i in range(2, 7)}
        if find_failure_cases:
            tests = {i: f"TriangularRandomSplitWorld[64]_({n})_diffuse_pairwise_test_{i}_split" for i in range(5, 7)}
    elif input_mode == 'qualitative':
        tests = {i: f'RandomSplitQualitativeWorld({n})_qualitative_test_{i}_split' for i in range(2, 6)}
    elif input_mode == 'stability_flat':
        tests = {i: f"RandomSplitWorld({n})_stability_flat_test_{i}_object" for i in range(4, 8)}
        # tests = {i: f"RandomSplitWorld({n})_stability_flat_test_{i}_object_i=31" for i in range(7, 8)}
    elif input_mode == 'robot_box':
        tests = {i: f"TableToBoxWorld({n})_robot_box_test_{i}_object" for i in range(3, 7)}
        if find_failure_cases:
            tests = {i: f"TableToBoxWorld(10)_robot_box_test_{i}_object_i=0" for i in range(6, 7)}
    else:
        raise NotImplementedError
    if find_failure_cases:
        tests = {k: v for i, (k, v) in enumerate(tests.items()) if i >= len(tests) - 2}
    return tests


def indie_runs():
    """ for developing and debugging """
    # check_data_graph(test_tasks[4])
    # check_data_graph('TriangularRandomSplitWorld(20)_test')

    ################ results for CoRL ###########################
    eval_10_kwargs = dict(tries=(10, 0), json_name='eval_N=10_K=10', save_log=False, visualize=True)
    eval_100_kwargs = dict(tries=(100, 0), json_name='eval', save_log=False, visualize=True)
    eval_100_render_kwargs = dict(tries=(100, 0), json_name='eval_2', save_log=True, visualize=True, render=True)

    ##  -----------------  triangle task

    test_10_tasks = {i: f"TriangularRandomSplitWorld[64]_(10)_diffuse_pairwise_test_{i}_split" for i in range(2, 7)}
    test_100_tasks = {i: f"TriangularRandomSplitWorld[64]_(100)_diffuse_pairwise_test_{i}_split" for i in range(2, 7)}
    # evaluate_model('gtqtyd8n', milestone=20, test_tasks=test_100_tasks, render=False, **eval_100_kwargs)
    # evaluate_model('dyz4eido', milestone=23, test_tasks=test_100_tasks, render=False, **eval_100_kwargs)

    ##  -----------------  qualitative task
    test_10_tasks = {i: f'RandomSplitQualitativeWorld(10)_qualitative_test_{i}_split' for i in range(2, 7)}
    test_100_tasks = {i: f'RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split' for i in range(2, 7)}
    evaluate_model('qsd3ju74', milestone=7, test_tasks=test_100_tasks, **eval_100_kwargs)
    # evaluate_model('r26wkb13', milestone=16, test_tasks=test_100_tasks, **eval_100_kwargs)

    ##  -----------------  3d stability tasks
    test_tasks = {i: f"RandomSplitWorld(100)_stability_flat_test_{i}_object" for i in range(5, 8)}
    # evaluate_model('ti639t00', milzestone=3, test_tasks=test_tasks, **eval_100_kwargs)
    # evaluate_model('bo02mwbw', milestone=11, test_tasks=test_tasks, **eval_100_kwargs)

    ## -----------------  3d stability tasks (TAMP pipeline)
    # for k in range(4, 7):
    #     test_tasks = {n: f"RandomSplitWorld(50)_stability_flat_test_{n}_object_all_n={n}_i={k}" for n in range(4, 6)}
    #     evaluate_model('ti639t00', milestone=3, tries=(1, 0), test_tasks=test_tasks, json_name=f'tamp_{k}', run_all=True)
    #     # evaluate_model('bo02mwbw', milestone=11, tries=(1, 0), test_tasks=test_tasks, json_name=f'tamp_{k}', run_all=True)

    ## ----------------- 3d packing tasks
    test_10_tasks = {i: f"TableToBoxWorld(10)_test_{i}_object" for i in range(2, 7)}
    test_100_tasks = {i: f"TableToBoxWorld(100)_robot_box_test_{i}_object" for i in range(3, 7)}
    # evaluate_model('0hhtsh6a', milestone=5, test_tasks=test_100_tasks, **eval_100_kwargs)
    # evaluate_model('qcjqkout', milestone=14, test_tasks=test_100_tasks, **eval_100_kwargs)

    # ## 3d packing tasks (TAMP pipeline)
    # for k in range(3, 9):
    #     test_tasks = {n: f"TableToBoxWorld(50)_robot_box_test_{n}_object_all_n={n}_i={k}" for n in range(2, 7)}
    #     evaluate_model('siovfil2', milestone=6, tries=(1, 0), test_tasks=test_tasks, json_name=f'tamp_{k}', run_all=True)  ## 0.1-0.14 sec per graph
    #     # evaluate_model('4xt8u4n7', milestone=20, tries=(1, 0), test_tasks=test_tasks, json_name=f'tamp_{k}', run_all=True)  ## 0.005-0.01 sec per graph

def eval_new_set():

    relation = "aligned_bottom"
    eval_relation = "aligned_bottom"

    EBM = False
    energy_wrapper = False

    small_task = True

    if small_task:
        n = 20
        visualize = True
    else:
        n = 200
        visualize = False

    if relation == "aligned_bottom":
        model_relation = [0]
        evaluate_relation = [0]
        end_idx = 11
        model_id = '5ckl6qnj'
        milestone = 11
    elif relation == "cfree":
        model_relation = [1]
        evaluate_relation = [1]
        end_idx = 11
        model_id = '6b95pov5'
        milestone = 13
    elif relation == "ccollide":
        model_relation = [2]
        if eval_relation == "cfree":
            evaluate_relation = [1]
        elif eval_relation == "ccollide":
            evaluate_relation = [2]

        end_idx = 11
        model_id = 'cvh3bsux'
        milestone = 13

    elif relation == "mixed_cfree":
        model_relation = [0, 1]
        evaluate_relation = [0, 1]
        end_idx = 11
        model_id = 'nlx9uk0b'
        # model_id = 'p507ks09'
        milestone = 13
    elif relation == "mixed_ccollide":
        model_relation = [0, 2]
        evaluate_relation = [0, 2]

        if eval_relation != relation:
            end_idx = 11
        else:
            end_idx = 3
        model_id = 'kx2ig70z'
        milestone = 17
    elif relation == "integrated_cfree&ccollide":
        model_relation = [0, 1, 2]

        if eval_relation != relation:
            end_idx = 11
        else:
            end_idx = 3
        evaluate_relation = [0, 1, 2]
        if EBM == "MALA":
            model_id = 'xpandggr'
            milestone = 8
        else:
            model_id = 'qnoni470'
            milestone = 13


    eval_10_kwargs = dict(tries=(10, 0), json_name='eval', save_log=False, visualize=visualize, test_set=True, return_history=False,
                        run_all=True, model_relation=model_relation, evaluate_relation=evaluate_relation, EBM=EBM, energy_wrapper=energy_wrapper, samples_per_step=3, eval_only=True)
 
    test_10_tasks = {i: f'RandomSplitSparseWorld({n})_tidy_test_{i}_split/{eval_relation}' for i in range(2, end_idx)}
    # evaluate_model('qsd3ju74', input_mode="aligned_bottom", milestone=7, test_tasks=test_5_tasks, **eval_5_kwargs)


    # evaluate_model('cegzrggl', input_mode="aligned_bottom", milestone=2, test_tasks=test_10_tasks, **eval_10_kwargs) # ULA, single
    # evaluate_model('mun5fyd7', input_mode="aligned_bottom", milestone=1, test_tasks=test_10_tasks, **eval_10_kwargs) # ULA, both
    # evaluate_model('vdl5asu6', input_mode="aligned_bottom", milestone=11, test_tasks=test_10_tasks, **eval_10_kwargs) # False, single
    # evaluate_model('m0l0yzfw', input_mode="aligned_bottom", milestone=10, test_tasks=test_10_tasks, **eval_10_kwargs) # False, both
    
    
    # evaluate_model('urnb8iua', input_mode="aligned_bottom", milestone=13, test_tasks=test_10_tasks, **eval_10_kwargs) # False, both
    # evaluate_model('trci7w1x', input_mode="aligned_bottom", milestone=11, test_tasks=test_10_tasks, **eval_10_kwargs) # False, both

   
    evaluate_model(model_id, input_mode="tidy", relation=relation, milestone=milestone, test_tasks=test_10_tasks, n_tasks=n, test_name=f"{eval_relation}_neg", **eval_10_kwargs) # False, both

def eval_tidy_set():

    model_relation = None
    relation = "symmetry"
    if model_relation is None:
        model_relation = relation
   
    small_task = True
    EBM = False
    energy_wrapper = False

    if small_task:
        if "all" in relation:
            n = 20
        else:
            n = 10
        visualize = True
    else:
        if "all" in relation:
            n = 200
        else:
            n = 100
        visualize = False

    '''
    1qqximol: aligned_vertical
    3g6710yo: centered
    3wvo65qz: all_composed_True
    4age4lp3: all_composed_False
    bzznpoa4: all_composed_False
    fbj3zu0m: in
    fjjurhul: aligned_vertical
    gt4vgdat: aligned_bottom #
    gu4jx6af: all_composed_False
    i12llyrt: centered
    i30ku15a: all_composed_False
    j5ag2ml6: next_to_edge
    ml53a6cg: aligned_bottom
    pgbbzyor: all_composed_True
    urdggpxs: on_top_of
    uun1oug9: in
    xojh01yn: next_to_edge
    yd2pp1o1: on_top_of
    '''

    if model_relation == "aligned_bottom":
        if energy_wrapper:
            model_id = '9t4vgdat'
            milestone = 5
        else:
            model_id = 'ml53a6cg'
            milestone = 20
    elif model_relation == "aligned_vertical":
        if energy_wrapper:
            model_id = '1qqximol'
            milestone = 3
        else:
            model_id = 'fjjurhul'
            milestone = 22
    elif model_relation == "centered":
        if energy_wrapper:
            model_id = '3g6710yo'
            milestone = 3
        else:
            model_id = "i12llyrt"
            milestone = 20
    elif model_relation == "on_top_of":
        if energy_wrapper:
            model_id = 'yd2pp1o1'
            milestone = 3
        else:
            model_id = "urdggpxs" 
            milestone = 20
    elif model_relation == "next_to_edge":
        if energy_wrapper:  
            model_id = 'xojh01yn'
            milestone = 3
        else:   
            model_id = "j5ag2ml6" 
            milestone = 20
    elif model_relation == "in":
        energy_wrapper = True
        model_id = 'uun1oug9'
        milestone = 20
        # if energy_wrapper:
        #     model_id = 'uun1oug9'
        #     milestone = 3
        # else:
        #     model_id = "fbj3zu0m" 
        #     milestone = 0
    elif model_relation == "all_composed_False":
        if energy_wrapper:
            model_id = "ovhwg5o0" #"tvvpzowp" 
            milestone = 29
        else:
            model_id = "eelutdfc" #"ekegfe6v"
            milestone = 29
        # if energy_wrapper:
        #     model_id = "tvvpzowp" #'4age4lp3' # "bzznpoa4"
        #     milestone = 7
        # else:
        #     model_id = "ekegfe6v"#"gu4jx6af" # "i30ku15a"
        #     milestone = 8
    elif model_relation == "all_composed_True":
        if energy_wrapper:
            model_id = "zgw7jzti" #'pgbbzyor'
            milestone = 5
        else:
            model_id = "7a135mwb" #"3wvo65qz"
            milestone = 6
    elif model_relation == "symmetry":
        model_id = "m6q1nwil"
        milestone = 24
    elif model_relation == "next_to":
        model_id = "foohkf13"
        milestone = 24

    
    # --- for testing
    train_task = f"RandomSplitSparseWorld(10000)_tidy_train/aligned_bottom"

    eval_10_kwargs = dict(tries=(10, 0), json_name='eval', save_log=False, visualize=visualize, test_set=True, return_history=False,
                        run_all=True, model_relation=relation, evaluate_relation=relation, EBM=EBM, 
                        energy_wrapper=energy_wrapper, samples_per_step=3, eval_only=True, train_task=train_task)
 
    test_10_tasks = {i: f'RandomSplitSparseWorld({n})_tidy_test_{i}_split/{relation}' for i in range(3, 11)}
   
    evaluate_model(model_id, input_mode="tidy", relation=relation, milestone=milestone, test_tasks=test_10_tasks, n_tasks=n, test_name=f"{relation}", **eval_10_kwargs) # False, both

def eval_customized_cases():
    ## wrapper True, composed True
    # wrapper True, composed False
    # wrapper True, composed False, ULA, 3
    # wrapper True, composed False, MALA, 3

    ## wrapper False, composed True
    # wrapper False, composed False
    # wrapper False, composed False, ULA, 3

    energy_wrapper = True

    EBM = "ULA"
    steps = 7

    model_relation = "all_composed_False"
    evaluate_relation = None

    if evaluate_relation is None:
        evaluate_relation = model_relation

    if model_relation == "all_composed_False":
        if energy_wrapper:
            model_id = "2w2vkfh2" #"ovhwg5o0" #"tvvpzowp" 
            milestone = 10
        else:
            model_id = "7w2orp88" #"eelutdfc" #"ekegfe6v"
            milestone = 14
    elif model_relation == "all_composed_True":
        if energy_wrapper:
            model_id = "zgw7jzti" 
            milestone = 5
        else:
            model_id = "7a135mwb" 
            milestone = 6

    # --- for testing
    train_task = f"RandomSplitSparseWorld(4)_tidy_train/study_table"

    eval_10_kwargs = dict(tries=(10, 0), json_name='eval', save_log=False, visualize=True, test_set=True, return_history=False,
                        run_all=True, model_relation=model_relation, evaluate_relation=evaluate_relation, EBM=EBM, 
                        energy_wrapper=energy_wrapper, samples_per_step=steps, eval_only=True, train_task=train_task)
 
    test_10_tasks = {1: f'RandomSplitSparseWorld(4)_tidy_train/study_table'}
   
    evaluate_model(model_id, input_mode="tidy", relation=model_relation, milestone=milestone, test_tasks=test_10_tasks, n_tasks="study_table", test_name="study_table", **eval_10_kwargs) # False, both

   



if __name__ == '__main__':
    # indie_runs()
    # eval_tidy_set()
    eval_customized_cases()