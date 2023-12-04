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


# Replace your telegram token here
TOKEN = 'bot_token'

import supervision
print("supervision.__version__:", supervision.__version__)
import cv2
import os
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


# settings
MODEL = "yolov8x.pt"
from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()

# # dict maping class_id to class_name
# CLASS_NAMES_DICT = model.model.names
# # class_ids of interest - person
# CLASS_ID = [0]

# SOURCE_VIDEO_PATH = "./pedestrian/passageway1-c2-resized.mp4"
# # create frame generator
# generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# # create instance of BoxAnnotator
# box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)
# # acquire first video frame
# iterator = iter(generator)
# frame = next(iterator)
# # model prediction on single frame and conversion to supervision Detections
# results = model(frame)
# detections = Detections(
#     xyxy=results[0].boxes.xyxy.cpu().numpy(),
#     confidence=results[0].boxes.conf.cpu().numpy(),
#     class_id=results[0].boxes.cls.cpu().numpy().astype(int)
# )
# # format custom labels
# labels = [
#     f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
#     for _, confidence, class_id, tracker_id
#     in detections
# ]
# # annotate and display frame
# frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

# print("detections")
# print(detections)

#show_frame_in_notebook(frame, (16, 16))

import logging
import requests
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, ConversationHandler, filters, CallbackContext
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
from count_people_on_image import count_people  # Import the function from the modified script.
from count_people_flow import count_people_flow  # Import the function from the modified script.

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Define states
CHOOSE, PROCESS_IMAGE, CHOOSE_ORIENTATION, UPLOAD_VIDEO = range(4)

async def start(update: Update, context: CallbackContext) -> int:
    reply_keyboard = [['Image', 'Video']]
    await update.message.reply_text(
        "Hi there! I am your People Counter Bot. 🚶‍♂️🚶‍♀️\n\n"
        "This bot can analyze both images and videos:\n"
        "1. **Image Analysis**: Send me an image, and I will count the number of people in it, providing you with a quick assessment of crowd size.\n"
        "2. **Video Analysis**: Send me a video, and I'll set up a virtual line (either vertical or horizontal). As the video plays, I will count the number of people crossing this line, useful for understanding movement patterns or traffic flow.\n\n"
        "Please select whether you want to upload an image or a video, and I'll guide you through the next steps.",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )
    return CHOOSE

async def choose(update: Update, context: CallbackContext) -> int:
    user_input = update.message.text.lower()
    
    if user_input == 'image':
        await update.message.reply_text(
            'Great! Please send me the image where you want to count people.'
        )
        return PROCESS_IMAGE
    elif user_input == 'video':
        reply_keyboard = [['Vertical', 'Horizontal']]
        await update.message.reply_text(
            'For videos, I count people crossing a line. '
            'Could you tell me if you want the line to be vertical or horizontal?',
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return CHOOSE_ORIENTATION
        # await update.message.reply_text('Please send me the video.')
        # return PROCESS_VIDEO

async def process_image(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text(
        'Thanks for the image! 🖼️ Give me a moment while I process it and count the people...'
    )
    
    # Extract the largest photo
    largest_photo = update.message.photo[-1]
    file_id = largest_photo.file_id
    
    # Get file path from Telegram
    get_file_url = f"https://api.telegram.org/bot{TOKEN}/getFile?file_id={file_id}"
    response = requests.get(get_file_url).json()
    
    # Check if request was successful
    if response['ok']:
        file_path = response['result']['file_path']
        
        # Construct URL for downloading the image
        download_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"

        # Download the image and save it locally
        response = requests.get(download_url)
        if response.status_code == 200:
            image_path = f"{file_id}.jpg"
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            # Pass the image to count_people_on_image.py script
            annotated_image, people_count = count_people(image_path, model)
            
            # Send the result back to user
            await update.message.reply_text(f'Number of people detected: {people_count}')
            cv2.imwrite('./temp/' + file_path, annotated_image)
            # If you want to send the annotated image back to user
            with open('./temp/' + file_path, 'rb') as img:
                await update.message.reply_photo(photo=img)
            
            # Optional: remove the image after processing to save space
            os.remove(image_path)
        else:
            await update.message.reply_text('Failed to download the image, please try smaller image size again')
    else:
        await update.message.reply_text('Failed to get the image path.')

    return ConversationHandler.END

async def process_video(update: Update, context: CallbackContext) -> int:
    # Code to process the video
    orientation = context.user_data.get('orientation', 'horizontal')
    await update.message.reply_text(
        f"Thanks for the video! 🎥 I'm setting up a {orientation} line and will count people crossing it. "
        "This might take a bit depending on the video length. Please be patient..."
    )
    # Your video processing code here
    print(update.message)
    video = update.message.video
    file_id = video.file_id
    
    # Get file path from Telegram
    get_file_url = f"https://api.telegram.org/bot{TOKEN}/getFile?file_id={file_id}"
    response = requests.get(get_file_url).json()
    print(response)
    if response['ok']:
        file_path = response['result']['file_path']
        download_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"

        # Download the image and save it locally
        response = requests.get(download_url)
        video_path = './temp/' + file_path
        if response.status_code == 200:
            with open(video_path, 'wb') as f:
                f.write(response.content)
        
        # Pass the video to your processing script/method
        # This is an example. You'll have to integrate your own video processing method.
        processed_video_paths, flow_count_text = count_people_flow(video_path, model, orientation)
        
        
        if(len(processed_video_paths) == 1):
            #send the text back to the user that process success and total number of video
            await update.message.reply_text(f'Processing success.')
        else:
            await update.message.reply_text(f'Processing success. Resulti video too large, spliting into {len(processed_video_paths)} videos.')

        for processed_video_path in processed_video_paths:
            # If you want to send the processed video back to the user
            with open(processed_video_path, 'rb') as vid:
                await update.message.reply_video(video=vid,read_timeout=60, write_timeout=60, connect_timeout=60)

        # Send the result back to the user (Assuming it's a count or something similar)
        result_text = ''
        for text in flow_count_text:
            result_text += text + '\n'

        await update.message.reply_text(result_text)
        # await update.message.reply_text(f'Result from video: {processed_video_path}, {number}, {orientation}')  # Modify as needed
        
        # # If you want to send the processed video back to the user
        # with open(processed_video_path, 'rb') as vid:
        #     await update.message.reply_video(video=vid)
        
        # # Optional: remove the video after processing to save space
        # os.remove(video_path)

    else:
        await update.message.reply_text('Failed to download the video, please try video size smaller than 20MB again')
        
    return ConversationHandler.END

async def choose_orientation(update: Update, context: CallbackContext) -> int:
    orientation = update.message.text.lower()
    context.user_data['orientation'] = orientation
    await update.message.reply_text(
        f'Alright, a {orientation} line it is. '
        'Please send me the video where you want to count people crossing this line.'
    )
    return UPLOAD_VIDEO

def main():

    application = Application.builder().token(TOKEN).write_timeout(600).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSE: [MessageHandler(filters.Regex('^(Image|Video)$'), choose)],
            UPLOAD_VIDEO: [MessageHandler(filters.VIDEO, process_video)],
            CHOOSE_ORIENTATION: [MessageHandler(filters.Regex('^(Vertical|Horizontal)$'), choose_orientation)],
            PROCESS_IMAGE: [MessageHandler(filters.PHOTO, process_image)],
        },
        fallbacks=[]
    )

    application.add_handler(conv_handler)

    max_retries = 5  # You can adjust this number
    backoff_time = 5  # Start with waiting 5 seconds. You can adjust this number

    for i in range(max_retries):
        try:
            # Run the bot until the user presses Ctrl-C
            application.run_polling(allowed_updates=Update.ALL_TYPES)
            break  # If polling runs without errors, break out of loop
            
        except Exception as e:
            logging.error(f"Error encountered: {e}")
            
            # Double the backoff time for each failure, up to a maximum of 120 seconds.
            backoff_time = min(backoff_time * 2, 120)
            logging.info(f"Retrying in {backoff_time} seconds...")
            
            time.sleep(backoff_time)
    else:
        logging.error("Max retries reached. Exiting the application.")
    # Run the bot until the user presses Ctrl-C
    # application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
