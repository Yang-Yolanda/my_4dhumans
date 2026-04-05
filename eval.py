import argparse
import os
from pathlib import Path
import traceback
from typing import Optional

import pandas as pd
import torch
from filelock import FileLock
from hmr2.configs import dataset_eval_config
from hmr2.datasets import create_dataset
from hmr2.utils import Evaluator, recursive_to
from tqdm import tqdm
from hmr2.configs import default_config
from hmr2.models.adapters import load_model_adapter
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from yacs.config import CfgNode as CN
# import torch
import torchvision
import os
# 强制只使用物理显卡 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--method', type=str, default='hmr2', help='method name: hmr2/romp/trace/bev/...')
    # parser.add_argument('--skip_errors', action='store_true', default=True, help='Skip bad samples/batches')
    parser.add_argument('--results_file', type=str, default='results/eval_regression.csv', help='Path to results file.')
    parser.add_argument('--dataset', type=str, default='3DPW-TEST', help='Dataset to evaluate') # choices=['H36M-VAL-P2', '3DPW-TEST', 'MPI-INF-TEST']
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of test samples to draw')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers used for data loading')
    parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    # 新增：允许在评估时跳过出错的样本/批次（如缺失图片）
    parser.add_argument('--skip_errors', action='store_true', default=False, help='Skip samples/batches that fail to load or process instead of aborting evaluation')

    args = parser.parse_args()

    # # Download and load checkpoints
    # # download_models(CACHE_DIR_4DHUMANS)
    # model, model_cfg = load_hmr2(args.checkpoint)

    # # Setup HMR2.0 model
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = model.to(device)
    # model.eval()

    # Download and load checkpoints
    # download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint) if args.method.lower() == 'hmr2' else (None, None)

    # Setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.method.lower() == 'hmr2':
        model, model_cfg = load_hmr2(args.checkpoint)
        model = model.to(device)
        model.eval()
    else:
        # 给非 HMR2 方法构造一个最小可用的 cfg（数据裁剪与归一化用 ImageNet 规范）
        model, model_cfg = None, default_config()
        # 输入分辨率：ROMP/TRACE/BEV 用 512；否则用 224
        if args.method.lower() in ('romp', 'trace', 'bev'):
            model_cfg.MODEL.IMAGE_SIZE = 512
        else:
            model_cfg.MODEL.IMAGE_SIZE = 224

        model_cfg.defrost()
        model_cfg.SMPL = CN(new_allowed=True)
        model_cfg.SMPL.NUM_BODY_JOINTS = 23   # SMPL身体关节数，用于计算num_pose
        model_cfg.EXTRA.PELVIS_IND = 0        # 已设置的骨盆索引，保留
        
        model_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
        model_cfg.MODEL.IMAGE_STD  = [0.229, 0.224, 0.225]
        # 骨盆索引缺省为 0（Evaluator 会用到）
        model_cfg.EXTRA.PELVIS_IND = 0
        model_cfg.freeze()

    # 通过适配层加载
    from hmr2.models.adapters import load_model_adapter
    adapter = load_model_adapter(args.method, args.checkpoint, device, preloaded_model=model)

    # Load config and run eval, one dataset at a time
    print('Evaluating on datasets: {}'.format(args.dataset), flush=True)
    for dataset in args.dataset.split(','):
        dataset_cfg = dataset_eval_config()[dataset]
        args.dataset = dataset
        # run_eval(model, model_cfg, dataset_cfg, device, args)
        # 原: run_eval(model, model_cfg, dataset_cfg, device, args)
        run_eval(adapter, model_cfg, dataset_cfg, device, args)

def run_eval(adapter, model_cfg, dataset_cfg, device, args):
    # Create dataset and data loader
    dataset = create_dataset(model_cfg, dataset_cfg, train=False)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    # List of metrics to log
    if args.dataset in ['H36M-VAL-P2','3DPW-TEST']:
        metrics = ['mode_re', 'mode_mpjpe']
        pck_thresholds = None
    if args.dataset in ['LSP-EXTENDED', 'POSETRACK-VAL', 'COCO-VAL']:
        metrics = ['mode_kpl2']
        pck_thresholds = [0.05, 0.1]

    # Setup evaluator object
    evaluator = Evaluator(
        dataset_length=int(1e8), 
        keypoint_list=dataset_cfg.KEYPOINT_LIST, 
        pelvis_ind=model_cfg.EXTRA.PELVIS_IND, 
        metrics=metrics,
        pck_thresholds=pck_thresholds,
    )

    # Go over the images in the dataset.
    if getattr(args, 'skip_errors', False):
        i = 0
        error = None
        data_iter = iter(dataloader)
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            except Exception as e:
                print(f"[WARN] Skipping sample/batch while loading: {e}")
                continue
            try:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = adapter.infer(batch)
                evaluator(out, batch)

                if i % args.log_freq == args.log_freq - 1:
                    evaluator.log()
                i += 1
            except Exception as e:
                print(f"[WARN] Skipping batch {i} during model/eval: {e}")
                continue
        # Final log after iteration completes
        evaluator.log()
    else:
        try:
            for i, batch in enumerate(tqdm(dataloader)):
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = adapter.infer(batch)
                evaluator(out, batch)
                if i % args.log_freq == args.log_freq - 1:
                    evaluator.log()
            evaluator.log()
            error = None
        except (Exception, KeyboardInterrupt) as e:
            traceback.print_exc()
            error = repr(e)
            i = 0

    # Append results to file
    metrics_dict = evaluator.get_metrics_dict()
    save_eval_result(args.results_file, metrics_dict, args.checkpoint, args.dataset, error=error, iters_done=i, exp_name=args.exp_name)


def save_eval_result(
    csv_path: str,
    metric_dict: float,
    checkpoint_path: str,
    dataset_name: str,
    # start_time: pd.Timestamp,
    error: Optional[str] = None,
    iters_done=None,
    exp_name=None,
) -> None:
    """Save evaluation results for a single scene file to a common CSV file."""

    timestamp = pd.Timestamp.now()
    exists: bool = os.path.exists(csv_path)
    exp_name = exp_name or Path(checkpoint_path).parent.parent.name

    # save each metric as different row to the csv path
    metric_names = list(metric_dict.keys())
    metric_values = list(metric_dict.values())
    N = len(metric_names)
    df = pd.DataFrame(
        dict(
            timestamp=[timestamp] * N,
            checkpoint_path=[checkpoint_path] * N,
            exp_name=[exp_name] * N,
            dataset=[dataset_name] * N,
            metric_name=metric_names,
            metric_value=metric_values,
            error=[error] * N,
            iters_done=[iters_done] * N,
        ),
        index=list(range(N)),
    )

    # Lock the file to prevent multiple processes from writing to it at the same time.
    lock = FileLock(f"{csv_path}.lock", timeout=10)
    with lock:
        df.to_csv(csv_path, mode="a", header=not exists, index=False)

if __name__ == '__main__':
    main()
