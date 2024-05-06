import cv2
from do_all import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open the input video
input_video_path = "/mnt/keremaydin/that-is-not-a-gun/polat_alemdar.mp4"
cap = cv2.VideoCapture(input_video_path)

# Get the total number of frames
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = 1024
frame_height = 1024

replace_func = ReplaceAll(
    device=device,
    will_replace=['pistol', 'gun', 'pistol in hand', 'gun in hand'],
    replace_with='orange carrot in hand'
)

# Define output video writer
output_video_path = "/mnt/keremaydin/that-is-not-a-gun/final_video.mp4"
out = cv2.VideoWriter(output_video_path, 
                      cv2.VideoWriter_fourcc(*'mp4v'), 
                      fps, 
                      (frame_width, frame_height))

frame_num = 0

# Process each frame
while True:

    ret, frame = cap.read()
    if not ret:
        break

    print(f'{frame_num+1}/{num_frames}')

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_reshaped = cv2.resize(frame_rgb, (1024, 1024))

    frame_pil = Image.fromarray(frame_reshaped)

    # Perform erosion on the frame
    new_frame = replace_func.replace(frame_pil)

    # Convert it into np array
    new_frame_np = np.array(new_frame)

    # Convert RGB to BGR
    bgr_image = cv2.cvtColor(new_frame_np, cv2.COLOR_RGB2BGR)

    # Write the eroded frame to the output video
    out.write(bgr_image)

    frame_num += 1

# Release resources
cap.release()
out.release()


