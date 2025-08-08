import os
import librosa
import pandas as pd
import torch
import io
import sys

from batdetect2 import api
import batdetect2.detector.compute_features as feats
import batdetect2.utils.detector_utils as du
import batdetect2.utils.audio_utils as au
from models.detection_interface import DetectionInterface
from utils.utils import gen_empty_df

import batdetect2.api as api

class BatCallDetector(DetectionInterface):
    """
    A class containing the bat detect model and feeding buzz model. The parameters of this class are explained in cfg.py 
    """
    def __init__(self, detection_threshold, spec_slices, chunk_size, time_expansion_factor, quiet, cnn_features):
        self.detection_threshold = detection_threshold
        self.spec_slices = spec_slices
        self.chunk_size = chunk_size
        self.time_expansion_factor = time_expansion_factor
        self.quiet = quiet
        self.cnn_features = cnn_features
        

    def get_name(self):
        return "BatDetectorMSDS"

    def _run_batdetect(self, audio_file)-> pd.DataFrame: #
        """
        Parameters:: 
            audio_file: a path containing the post-processed wav file.

        Returns:: a pd.Dataframe containing the bat calls detections
        """
        config = api.get_config(detection_threshold=self.detection_threshold,
                                spec_slices = self.spec_slices,
                                chunk_size = self.chunk_size,
                                time_expansion_factor = self.time_expansion_factor,
                                quiet = self.quiet,
                                cnn_features = self.cnn_features)
        model = api.MODEL
        device = api.DEVICE

        # Suppress output from this call
        text_trap = io.StringIO()
        sys.stdout = text_trap

        # store temporary results here
        predictions = []
        spec_feats = []
        cnn_feats = []
        spec_slices = []

        # Get original sampling rate
        file_samp_rate = librosa.get_samplerate(audio_file)
        orig_samp_rate = file_samp_rate * (config.get("time_expansion") or 1)

        # load audio file
        sampling_rate, audio_full = au.load_audio(
            audio_file,
            time_exp_fact=config.get("time_expansion", 1) or 1,
            target_samp_rate=config["target_samp_rate"],
            scale=config["scale_raw_audio"],
            max_duration=config.get("max_duration"),
        )

        # loop through larger file and split into chunks
        # TODO: fix so that it overlaps correctly and takes care of
        # duplicate detections at borders
        for chunk_time, audio in du.iterate_over_chunks(
            audio_full,
            sampling_rate,
            config["chunk_size"],
        ):
            # Run detection model on chunk
            pred_nms, features, spec = du._process_audio_array(
                audio,
                sampling_rate,
                model,
                config,
                device,
            )
            num_rawdets = pred_nms['start_times'].shape[0]

            raw_dets = pd.DataFrame()
            for key in pred_nms.keys(): 
                if key != 'class_probs':                 
                    raw_dets[key] = pred_nms[key]

            class_probs = []
            for i in range(num_rawdets):
                class_probs_for_det = pred_nms['class_probs'][:,i]
                class_probs.append(class_probs_for_det)
            raw_dets['class_probs'] = class_probs
            raw_dets['chunk_time'] = [chunk_time]*len(raw_dets)
            inscope_rawdets = raw_dets[raw_dets['end_times']<=audio.shape[0]/sampling_rate]
            outofscopes_rawdets = raw_dets[raw_dets['end_times']>=audio.shape[0]/sampling_rate]

            inscope_features = features[inscope_rawdets.index,:]
            inscope_pred_nms = dict()
            for key in pred_nms.keys():
                if key == 'class_probs':
                    inscope_pred_nms[key] = pred_nms[key][:,inscope_rawdets.index]
                else:
                    inscope_pred_nms[key] = pred_nms[key][inscope_rawdets.index]
            # convert to numpy
            spec_np = spec.detach().cpu().numpy().squeeze()

            # add chunk time to start and end times
            inscope_pred_nms["start_times"] += chunk_time
            inscope_pred_nms["end_times"] += chunk_time

            predictions.append(inscope_pred_nms)

            # extract features - if there are any calls detected
            if inscope_pred_nms["det_probs"].shape[0] == 0:
                continue

            if config["spec_features"]:
                spec_feats.append(feats.get_feats(spec_np, inscope_pred_nms, config))

            if config["cnn_features"]:
                cnn_feats.append(inscope_features[0])

            if config["spec_slices"]:
                # FIX: This is not currently working. Returns empty slices
                spec_slices.extend(feats.extract_spec_slices(spec_np, inscope_pred_nms))

        # Merge results from chunks
        predictions, spec_feats, cnn_feats, spec_slices = du._merge_results(
            predictions,
            spec_feats,
            cnn_feats,
            spec_slices,
        )

        # convert results to a dictionary in the right format
        model_output = du.convert_results(
            file_id=os.path.basename(audio_file),
            time_exp=config.get("time_expansion", 1) or 1,
            duration=audio_full.shape[0] / float(sampling_rate),
            params=config,
            predictions=predictions,
            spec_feats=spec_feats,
            cnn_feats=cnn_feats,
            spec_slices=spec_slices,
            nyquist_freq=orig_samp_rate / 2,
        )

        # summarize results
        if not config["quiet"]:
            du.summarize_results(model_output, predictions, config)
        
        # Restore stdout
        sys.stdout = sys.__stdout__

        annotations = model_output['pred_dict']['annotation']

        out_df = gen_empty_df()
        if annotations:
            out_df = pd.DataFrame.from_records(annotations) 
        return out_df