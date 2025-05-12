import torch
from os import listdir
import numpy as np
from os.path import isfile, join
import h5py
import json
import argparse
import cv2
import os
import logging
from utils.utils import get_paths, setup_logging
from evaluation.evaluation_metrics import evaluate_summary
from model.layers.summarizer import xLSTM
from .generate_summary import generate_summary


setup_logging()

def load_video_data(data_path, video):
    """Load video data from the dataset."""
    with h5py.File(data_path, "r") as hdf:
        frame_features = torch.Tensor(np.array(hdf[f"{video}/features"])).view(-1, 1024)
        user_summary = np.array(hdf[f"{video}/user_summary"])
        sb = np.array(hdf[f"{video}/change_points"])
        n_frames = np.array(hdf[f"{video}/n_frames"])
        positions = np.array(hdf[f"{video}/picks"])
        video_name = None
        if "video_name" in hdf[f"{video}"]:
            video_name = str(np.array(hdf[f"{video}/video_name"]).astype(str, copy=False))
    return frame_features, user_summary, sb, n_frames, positions, video_name


def save_video_frames(video_summaries, video_names, split_id, dataset, save_frames=True):
    """Save summarized video frames and videos."""
    for video, summary_indices in video_summaries.items():
        video_name = video if dataset == 'TVSum' else video_names[video].replace(" ", "_")
        frames_folder = os.path.join(f"data/summarized_frames/{dataset}/{video}")
        os.makedirs(frames_folder, exist_ok=True)

        first_frame_path = os.path.join(f"data/frames/{dataset}/{video_name}", 'img_00001.jpg')
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            logging.error(f"First frame of video {video_name} not found.")
            continue

        frame_height, frame_width, _ = first_frame.shape
        total_frame_quantity = len(summary_indices)
        generated_frame_quantity = 0

        video_path = f'{video}_{video_name}_summary.mp4'
        if os.path.exists(video_path):
            logging.info(f"Video {video_path} already exists. Skipping...")
            continue

        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60.0, (frame_width, frame_height))
        logging.info(f"Processing video: {video_name} - SPLIT {split_id}")

        for index, is_selected in enumerate(summary_indices):
            if is_selected == 1:
                frame_path = os.path.join(f"data/frames/{video_name}", f'img_{index+1:05d}.jpg')
                frame = cv2.imread(frame_path)
                if frame is not None:
                    generated_frame_quantity += 1
                    out.write(frame)
                    if save_frames:
                        frame_save_path = os.path.join(frames_folder, f'img_{index+1:05d}.jpg')
                        cv2.imwrite(frame_save_path, frame)

        logging.info(f"Original frame quantity for {video_name}: {total_frame_quantity}")
        logging.info(f"Summarized frame quantity for {video_name}: {generated_frame_quantity}")
        logging.info(f"Generated frames percentage: {(generated_frame_quantity / total_frame_quantity) * 100:.2f}%")
        out.release()
        logging.info(f"Saved summarized video frames in {frames_folder}")


def run_inference(model, data_path, keys, eval_method, save_summary, verbose=False):
    """Run inference on the dataset."""
    model.eval()
    video_fscores = []
    video_summaries = {}
    video_names = {}
    video_fscore_dict = {}
    summe = data_path.split('/')[1] == 'SumMe'

    for video in keys:
        video_number = int(video.split('_')[1])
        if summe and video_number > 25:
            logging.info(f"Skipping video {video}...")
            continue

        frame_features, user_summary, sb, n_frames, positions, video_name = load_video_data(data_path, video)

        with torch.no_grad():
            scores, _ = model(frame_features)
            scores = scores.squeeze(0).cpu().numpy().tolist()
            summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
            f_score = evaluate_summary(summary, user_summary, eval_method)

            video_fscores.append(f_score)
            video_summaries[video] = summary
            video_fscore_dict[video] = f_score

            if verbose:
                logging.debug(f"Summary for video {video}: {summary}")
                logging.debug(f"F-score for video {video_name}: {f_score:.2f}%")

            if summe:
                video_names[video] = video_name

            if save_summary:
                summary_json = {str(i): int(frame) for i, frame in enumerate(summary)}
                json_filename = f"{video}_summary.json"
                with open(json_filename, "w") as json_file:
                    json.dump(summary_json, json_file, indent=4)
                logging.info(f"Summary exported to {json_filename}")

    mean_fscore = np.mean(video_fscores)
    logging.info(f"Mean F-score: {mean_fscore:.2f}%")

    if summe:
        return mean_fscore, video_summaries, video_names
    else:
        return mean_fscore, video_summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='SumMe', help="Dataset to be used. Supported: {SumMe, TVSum}")
    parser.add_argument("--test_dataset", type=str, default='SumMe', help="Dataset to be used for testing. Supported: {SumMe, TVSum}")
    parser.add_argument("--table", type=str, default='4', help="Table to be reproduced. Supported: {3, 4}")
    parser.add_argument("--best_fscore_only", type=bool, default=True, help="Generate summarized video for the best F-score only.")
    parser.add_argument("--sum_split", type=int, default=None, help="Which split to summarize.")
    parser.add_argument("--verbose", type=int, default=False, help="Debugging logs.")
    parser.add_argument("--save_video", type=int, default=False, help="Whether to save the videos.")
    parser.add_argument("--save_summary", type=int, default=False, help="Whether to save the summary.")

    args = vars(parser.parse_args())
    dataset = args["dataset"]
    test_dataset = args["test_dataset"]
    best_fscore_only = args["best_fscore_only"]
    sum_split = args["sum_split"]
    verbose = args["verbose"]
    save_video = args["save_video"]
    save_summary = args["save_summary"]

    eval_metric = 'avg' if dataset.lower() == 'tvsum' else 'max'
    all_fscores = []

    # TODO: re-implement test_dataset
    logging.info(f"Running inference for {dataset} dataset, testing with {test_dataset} dataset")

    for split_id in range(5):
        model_path = f"Summaries/xLSTM/{dataset}/models/split{split_id}"
        model_file = [f for f in listdir(model_path) if isfile(join(model_path, f))]
        paths = get_paths(dataset)
        split_file = paths['split']

        with open(split_file) as f:
            data = json.loads(f.read())
            test_keys = data[split_id]["test_keys"]
        
        dataset_path = paths['dataset']
        trained_model = xLSTM(input_size=1024, output_size=1024, num_segments=4, hidden_dim=512, num_layers=2, dropout=0.2)
        trained_model.load_state_dict(torch.load(join(model_path, model_file[-1])))

        if dataset == 'SumMe':
            fscore, video_summaries, video_names = run_inference(trained_model, dataset_path, test_keys, eval_metric, save_summary, verbose)
        else:
            fscore, video_summaries = run_inference(trained_model, dataset_path, test_keys, eval_metric, save_summary, verbose)

        all_fscores.append(fscore)

        if save_video and not best_fscore_only:
            save_video_frames(video_summaries, video_names, split_id, dataset)

    logging.info(f"Overall Mean F-score: {np.mean(all_fscores):.2f}%")


if __name__ == "__main__":
    main()