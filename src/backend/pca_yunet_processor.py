import cv2
import numpy as np
from typing import Union, List, Tuple
import time
class YuNetProcessor:
    """
    A class to encapsulate the YuNet model loading, configuration,
    and frame processing logic for real-time face detection.

    This class applies Object-Oriented Programming (OOP) principles like:
    - Encapsulation: Bundling data (model, thresholds, input size) and methods
      that operate on that data within a single unit.
    - Abstraction: Hiding the complex internal details of model loading and
      frame processing, exposing only necessary methods for interaction.
    """

    def __init__(self, model_path: str = None, initial_confidence_threshold: float = 0.9, 
                 input_size: Tuple[int, int] = (640, 640)):
        """
        Constructor for the YuNetProcessor class.
        Initializes the YuNet model, sets up the device (CPU by default), and defines initial parameters.

        Args:
            model_path (str): Path to the YuNet model weights (e.g., 'face_detection_yunet_2023mar.onnx').
                            If None, uses the default model from OpenCV.
            initial_confidence_threshold (float): The default confidence threshold for detections.
                                                Detections with confidence below this will be filtered out.
            input_size (Tuple[int, int]): The target image size (width, height) for model inference.
                                        YuNet typically uses square or rectangular inputs (e.g., 320x320 or 640x480).
                                        Frames will be resized to this dimension before being fed to the model.
        """
        # Use default model if no path is provided
        if model_path is None:
            model_path = "face_detection_yunet_2023mar.onnx"

        print(f"Loading YuNet model from {model_path}...")
        # Load the YuNet model using OpenCV's FaceDetectorYN
        self.model = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=initial_confidence_threshold,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        
        # Store the confidence threshold and image size as private attributes
        self._confidence_threshold = initial_confidence_threshold
        self._input_size = input_size
        
        print("YuNet model loaded and initialized.")

    def _pad_to_divisible(self, image: np.ndarray, stride: int = 32) -> Tuple[np.ndarray, int, int]:
        """
        A private helper method to pad an image.
        Ensures that the image dimensions are divisible by the given stride,
        which is often a requirement for deep learning models.

        Args:
            image (np.ndarray): The input image as a NumPy array (H, W, C).
            stride (int): The stride value (e.g., 32 for YuNet).

        Returns:
            Tuple[np.ndarray, int, int]: A tuple containing:
                - padded_image (np.ndarray): The image with added padding.
                - pad_h (int): The amount of padding added to the height.
                - pad_w (int): The amount of padding added to the width.
        """
        h, w = image.shape[:2]
        new_h = ((h + stride - 1) // stride) * stride
        new_w = ((w + stride - 1) // stride) * stride
        
        pad_h = new_h - h
        pad_w = new_w - w

        padded_image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded_image, pad_h, pad_w

    def set_confidence_threshold(self, threshold: float):
        """
        Public method to set (update) the confidence threshold for face detection.
        This allows external components to dynamically change the model's behavior.

        Args:
            threshold (float): The new confidence threshold (must be between 0.0 and 1.0).
        
        Raises:
            ValueError: If the provided threshold is outside the valid range.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self._confidence_threshold = threshold
        self.model.setScoreThreshold(threshold)
        print(f"YuNet confidence threshold updated to: {self._confidence_threshold:.2f}")

    def get_confidence_threshold(self) -> float:
        """
        Public method to get the current confidence threshold.
        """
        return self._confidence_threshold

    def set_input_size(self, input_size: Tuple[int, int]):
        """
        Public method to set (update) the target input size for model inference.
        This can be useful if you want to change the input resolution of the model.

        Args:
            input_size (Tuple[int, int]): New target input size (width, height).
        
        Raises:
            ValueError: If the input_size is not a valid tuple of positive integers.
        """
        if not isinstance(input_size, tuple) or len(input_size) != 2 or not all(isinstance(x, int) and x > 0 for x in input_size):
            raise ValueError("input_size must be a tuple of two positive integers (width, height).")
        self._input_size = input_size
        self.model.setInputSize(input_size)
        print(f"YuNet inference input size updated to: {self._input_size}")

    def get_input_size(self) -> Tuple[int, int]:
        """
        Public method to get the current inference input size.
        """
        return self._input_size

    def process_frames(self, frames: Union[np.ndarray, List[np.ndarray]]) -> Union[Tuple[np.ndarray, List[dict]], Tuple[List[np.ndarray], List[List[dict]]]]:
        """
        Main public method to process a single frame or a list of frames using the YuNet model.
        It orchestrates the preprocessing, actual model inference, and post-processing.

        Args:
            frames (Union[np.ndarray, List[np.ndarray]]): A single image or a list of images (H, W, C).

        Returns:
            Union[Tuple[np.ndarray, List[dict]], Tuple[List[np.ndarray], List[List[dict]]]]: Processed frame(s) and detection(s).
        """
        try:
            total_start_time = time.time()
            pre_start = time.time()

            is_batch_input = isinstance(frames, list)
            frames_list = frames if is_batch_input else [frames]

            processed_frames = []
            detections_list = []

            for frame in frames_list:
                if not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) != 3:
                    print(f"Invalid frame detected: {frame}")
                    processed_frames.append(np.zeros_like(frame) if isinstance(frame, np.ndarray) else np.zeros((480, 640, 3), dtype=np.uint8))
                    detections_list.append([])
                    continue

                original_frame_sizes = (frame.shape[1], frame.shape[0])
                if self._input_size and (frame.shape[1] != self._input_size[0] or frame.shape[0] != self._input_size[1]):
                    frame_resized = cv2.resize(frame, self._input_size, interpolation=cv2.INTER_AREA)
                else:
                    frame_resized = frame

                frame_padded, _, _ = self._pad_to_divisible(frame_resized, stride=32)
                preprocess_time = time.time() - pre_start

                # Inference directly with detect
                infer_start = time.time()
                faces = self.model.detect(frame_padded)  # Returns (faces, scores) or (None, None) if no detection
                inference_time = time.time() - infer_start

                post_start = time.time()
                output_frame = frame.copy()
                detections = []

                if faces[1] is not None:  # faces[1] contains the detection results
                    for face in faces[1]:
                        x1, y1, w_box, h_box, conf = map(float, face[:5])
                        x2, y2 = x1 + w_box, y1 + h_box

                        original_w, original_h = original_frame_sizes
                        scale_x = original_w / self._input_size[0]
                        scale_y = original_h / self._input_size[1]
                        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

                        x1, x2 = np.clip([x1, x2], 0, original_w)
                        y1, y2 = np.clip([y1, y2], 0, original_h)

                        detections.append({
                            "class_name": "Face",
                            "confidence": conf,
                            "box": [x1, y1, x2, y2]
                        })

                        # Draw the bounding box and label
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Face {conf:.2f}"
                        text_y_pos = y1 - 10 if y1 - 10 > 0 else y1 + 20
                        cv2.putText(output_frame, label, (x1, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                processed_frames.append(output_frame)
                detections_list.append(detections)

                postprocess_time = time.time() - post_start
                total_time = time.time() - total_start_time
                # print(f"[TIME] Pre: {preprocess_time:.3f}s | Infer: {inference_time:.3f}s | Post: {postprocess_time:.3f}s | Total: {total_time:.3f}s")

            return (processed_frames[0], detections_list[0]) if not is_batch_input else (processed_frames, detections_list)

        except Exception as e:
            print(f"Error in YuNetProcessor.process_frames: {e}")
            import traceback
            traceback.print_exc()
            if is_batch_input:
                return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in frames_list], [[] for _ in frames_list]
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8), []