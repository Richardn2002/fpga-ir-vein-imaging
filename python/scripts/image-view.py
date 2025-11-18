import numpy as np
import cv2
import sys


def read_frame(filename, width, height):
    # Calculate the frame size in bytes (width * height * 2)
    frame_size = width * height * 2

    # Open the file in binary read mode ('rb')
    with open(filename, "rb") as f:
        # Read the entire file content into a bytes object
        # If the file contains multiple frames, you would need to read in chunks/frames
        raw_data = f.read(frame_size)

    if len(raw_data) != frame_size:
        print(
            f"Warning: Read only {len(raw_data)} bytes, expected {frame_size} bytes for a single frame."
        )

    # Convert the bytes object to a numpy array of unsigned 8-bit integers
    array = np.frombuffer(raw_data, dtype=np.uint8)

    return array


def display_yuyv_frame(yuyv_array, width, height):
    """
    Converts a YUYV numpy array to a BGR image for display with OpenCV.

    Args:
        yuyv_array (numpy.ndarray): The raw YUYV data.
        width (int): The width of the frame.
        height (int): The height of the frame.
    """
    # Reshape the array to the correct dimensions for OpenCV YUYV conversion
    # OpenCV expects the data to be in a specific matrix layout for conversion
    yuyv_shaped = yuyv_array.reshape((height, width, 2))

    # Convert from YUYV to BGR (OpenCV's default color space)
    bgr_image = cv2.cvtColor(yuyv_shaped, cv2.COLOR_YUV2BGR_YUYV)
    bgr_image = cv2.resize(
        bgr_image, (width * 4, height * 4), interpolation=cv2.INTER_NEAREST
    )

    # Display the image
    cv2.imshow("YUYV Frame (BGR conversion)", bgr_image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()


def display_rgb565_frame(frame, width, height):
    bgr_image = []
    for pixel_idx in range(width * height):
        first_byte = frame[2 * pixel_idx]
        second_byte = frame[2 * pixel_idx + 1]
        bgr_image.extend(
            [
                (second_byte & 0b11111) * 255 // 31,  # B
                (((first_byte & 0b111) << 3) + (second_byte >> 5)) * 255 // 63,  # G
                (first_byte >> 3) * 255 // 31,  # G
            ]
        )
    bgr_image = np.array(bgr_image, dtype=np.uint8)
    bgr_image = bgr_image.reshape((height, width, 3))
    bgr_image = cv2.resize(
        bgr_image, (width * 4, height * 4), interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow("BGR Frame", bgr_image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()


if len(sys.argv) < 4:
    print("Usage: python image-view.py <data file> <width> <height>")
    sys.exit(-1)

filename = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])

frame = read_frame(filename, width, height)
# display_rgb565_frame(frame, width, height)
display_yuyv_frame(frame, width, height)
