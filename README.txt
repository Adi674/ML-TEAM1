# People Tracking using YOLOv8

## Overview

This project utilizes the YOLOv8 deep learning model to detect and track people in a given video. A unique ID is assigned to each detected person using basic tracking logic.

## Features

- Uses **YOLOv8x** for high-accuracy person detection.
- Assigns **unique IDs** to detected people.
- Displays bounding boxes and IDs on each person.
- Shows the total number of people detected in each frame.
- Outputs a processed video with tracking annotations.

## Requirements

Ensure you have the following dependencies installed before running the script:

```bash
pip install ultralytics opencv-python uuid
```

## Usage

1. Place your input video file in the specified directory.
2. Modify the `video_path` variable to the correct video file path.
3. Run the script to process the video and generate an output file.

## How It Works

1. **Load YOLOv8 Model**: The model is loaded using the `ultralytics` library.
2. **Process Video Frames**: Each frame is read, and the YOLO model detects people.
3. **Assign Unique IDs**: Every detected person gets a UUID-based identifier.
4. **Draw Bounding Boxes & Labels**: Each detected person is marked with a rectangle and ID.
5. **Save Processed Video**: The annotated video is saved as an output file.

## Expected Output

- The processed video will display people with unique IDs and a counter showing the number of detected individuals.
- The output file is saved as `output_tracked_v8.mp4`.

## Notes

- The YOLOv8x model is a **heavy** version and may require a GPU for efficient processing.
- The tracking logic in this script does **not** implement advanced tracking algorithms like DeepSORT.
- Ensure the input video has clear visibility for better detection accuracy.

## Author

This project was developed using YOLOv8 for people tracking in videos.
