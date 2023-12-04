from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator


from typing import List

import numpy as np


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

from typing import Tuple
from ultralytics import YOLO
from custom_line_counter import CustomLineCounterAnnotator, CustomLineCounter
from tqdm.notebook import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2


def count_people_flow(SOURCE_VIDEO_PATH: np.ndarray, model: YOLO, line_orientation: str) -> Tuple[str, int]:
    # dict maping class_id to class_name
    CLASS_NAMES_DICT = model.model.names
    # class_ids of interest - person
    CLASS_ID = [0]

    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    LINE_START = Point(0, 0)
    LINE_END = Point(0, 0)

    video_width = video_info.width
    video_height = video_info.height

    if(line_orientation == "vertical"):
        LINE_START = Point(video_width / 2, 0)
        LINE_END = Point(video_width / 2, video_height)
    elif(line_orientation == "horizontal"):
        LINE_START = Point(0, video_height / 2)
        LINE_END = Point(video_width, video_height / 2)

    print(f"Line start: {LINE_START}")
    print(f"Line end: {LINE_END}")

    # create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    # create frame generator
    generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

    # create LineCounter instance
    line_counter = CustomLineCounter(start=LINE_START, end=LINE_END, line_orientation=line_orientation)
    # line_counter = LineCounter(start=LINE_START, end=LINE_END)

    # create instance of BoxAnnotator and LineCounterAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)

    line_annotator = CustomLineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)
    # line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

    source_video_basename = os.path.basename(SOURCE_VIDEO_PATH)
    output_video_name = os.path.splitext(source_video_basename)[0] + "_processed.mp4"
    TARGET_VIDEO_PATH = os.path.join('./temp/', output_video_name)

    # open target video file

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, video_info.fps, (video_info.width, video_info.height))

    # with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    frame_counter = 0
    # loop over video frames
    for frame in tqdm(generator, total=video_info.total_frames):
        frame_counter += 1

        # Check if 100 frames have been processed
        # if frame_counter >= 100:
        #     break
        
        # model prediction on single frame and conversion to supervision Detections
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # tracking detections
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # format custom labels
        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        # updating line counter
        line_counter.update(detections=detections)
        # annotate and display frame
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        line_annotator.annotate(frame=frame, line_counter=line_counter)

        out.write(frame)
    out.release()
            # sink.write_frame(frame)
    print(line_counter.get_texts())

    video_parts = []
    if os.path.getsize(TARGET_VIDEO_PATH) > 20 * 1_000_000:  # If file size is greater than 20MB
        video_parts = split_video(TARGET_VIDEO_PATH)
        print(f"Video was split into {len(video_parts)} parts: {video_parts}")
    else:
        print(f"Processed video saved at: {TARGET_VIDEO_PATH}")    
        video_parts = [TARGET_VIDEO_PATH]

    return video_parts, line_counter.get_texts()

def split_video(video_path: str, max_size_MB: float = 20.0) -> List[str]:
    MB_IN_BYTES = 1_000_000  # Defining MB as 1 million bytes
    max_size_bytes = max_size_MB * MB_IN_BYTES
    video_dir, video_filename = os.path.split(video_path)
    video_basename, video_ext = os.path.splitext(video_filename)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    segments = []
    segment_count = 0
    out = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a new segment if needed
        if out is None:
            segment_count += 1
            segment_path = os.path.join(video_dir, f"{video_basename}_part{segment_count}{video_ext}")
            segments.append(segment_path)
            out = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))

        out.write(frame)

        # Check the size of the current segment. If it's near the limit, close this segment and create a new one
        if os.path.getsize(segment_path) >= (0.9 * max_size_bytes):  # Checking at 90% of max size to be safe
            out.release()
            out = None

    if out:
        out.release()
    cap.release()

    return segments

test_script = False
MODEL = "yolov8x.pt"
from ultralytics import YOLO
if(test_script): 

    model = YOLO(MODEL)
    model.fuse()

    # video_path = './temp/pedestrian/passageway1-c2-001.avi'
    video_path = './pedestrian/passageway1-c2-001.avi'
    # vertical or horizontal
    orientation = 'vertical'

    processed_video_path, number = count_people_flow(video_path, model, orientation)

    print(f"Number of people: {number}")

# print(os.path.getsize("./temp/output_mp4v_x15.mp4"))