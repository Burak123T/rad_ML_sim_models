

import cv2
import numpy as np
import sys
import keras
import os

import time
import ctypes

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

librad = ctypes.CDLL('/usr/local/lib/librad_logger.so')
print("\n---------------------------")
print("| Loaded librad_logger.so |")
print("---------------------------\n")

BIN_OUTPUT_PATH = "../../bin_outputs/sar_preprocessed_input_data_keras.bin"

rad_log_start = librad.log_start

# int event_id, const char *event_name, const char *event_type, const char *event_data
arg_types = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
rad_log_start.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]

# Hard-coded slowdown time threshold
SLOW_THRESHOLD = 0.025

def preprocess_data(image_path):
    try:
        print("\033[96m" + "Reading input for preprocessing:" + "\033[0m \033[1m" + "{} \033[0m".format(image_path))

        # Load provided image
        image_unproc = cv2.imread(image_path)
        if image_unproc is None:
            raise ValueError(f"Could not open or read image file: {image_path}")
        else:
            print("\t- loaded image: {}".format(image_path))

        img_convert = cv2.cvtColor(image_unproc, cv2.COLOR_BGR2RGB)
        resize_img = cv2.resize(img_convert, (256, 256))
        image = resize_img.astype(np.float32) / 255.0
        print("\t- Scaled to: 0, 1")

        # Save as raw binary data (float32)
        image.astype(np.float32).tofile(BIN_OUTPUT_PATH)
        print("\033[96m" + f"Preprocessed image saved to:" + "\033[0m \033[92m" + f"{BIN_OUTPUT_PATH}" + "\033[0m \n")

        preprocess_result = rad_log_start(20, b'preprocess_data', b'inference data pre-processing', str(image).encode())
        print("\033[94mrad_logger:\033[0m preprocessing result logged for event id -> ", preprocess_result, "\n\n\n")

        return image

    except Exception as e:
        print("\033[91m" + f"Error during preprocessing: {e} \033[0m\n")

        preprocess_result = rad_log_start(11, b'preprocess_data', b'inference data pre-processing error!', str(e).encode())
        print("\033[94mrad_logger:\033[0m preprocessing result logged for event id -> ", preprocess_result, "\n\n\n")

        sys.exit(1)

def run_inference(model, prep_image):
    try:
        print("\033[96m" + "Predicting..." + "\033[0m")

        inf_data = prep_image.reshape((1,256,256,3))
        #print("\t- Pre-processed image reshapen")
        #print(f"\t- Shape of inf_data: {inf_data.shape}")

        model.predict(inf_data)

        run_inference_result = rad_log_start(40, b'run_inference', b'inference run', b'none')
        print("\033[94mrad_logger:\033[0m run inference result logged for event id -> ", run_inference_result)
        
        #print("outputs[0] -> ", outputs[0])
        #print("outputs[0].host -> ", outputs[0].host)
    except Exception as e:
        print("\n" + "\033[91m" + f"Error during inference stage: \n{e}" + "\033[0m \n")
        
        run_inference_result = rad_log_start(41, b'run_inference', b'error during inference run!', str(e).encode())
        print("\033[94mrad_logger:\033[0m run inference result logged for event id -> ", run_inference_result)

        exit()

def post_proc_results(pred_output):

    print("\033[96m" + "Running post-processing (no saving)" + "\033[0m")

    try:
        x = 256 * 256
        if x != pred_output.size:
            print("\033[91m" + f"Mismatch for output size: {pred_output.size} \033[0m\n")
        else:
            print("\033[96m" + f"Correct size \033[0m\n")
            postproc_result = rad_log_start(50, b'post_proc_results', b'post-process infered results', str(x).encode())
            print("\033[94mrad_logger:\033[0m post-processing results logged for event id -> ", postproc_result)
    except Exception as e:
        print("\033[91m" + f"Error during post-processing results:\n {e}" + "\033[0m \n")
        
        postproc_result = rad_log_start(51, b'post_proc_results', b'post-processsing error', str(e).encode())
        print("\033[94mrad_logger:\033[0m post-processing results logged for event id -> ", postproc_result)

        exit()

# Check if inference time above threshold
def check_slowdown(total_inf_time):
    if total_inf_time > SLOW_THRESHOLD:
        # log time slowdown
        log_slow_time = rad_log_start(80, b'Inference time below threshold', b'inference slowdown', str(total_inf_time).encode())
        print("\033[91mrad_logger:\033[0m inference slowdown logged for event id -> ", log_slow_time)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: <keras_model_path> <input_image_path> <optional_number_of_iterations>")
        sys.exit(1)

    if len(sys.argv) > 4:
        print("Usage: <keras_model_path> <input_image_path> <optional_number_of_iterations>")
        sys.exit(1)

    keras_model_path = sys.argv[1]
    input_image_path = sys.argv[2]
    prep_image = preprocess_data(input_image_path)
    model = keras.models.load_model(keras_model_path)

    for i in range(5):
        warmup_start_time = time.time()
        print(f"---------- WARMUP {i + 1} ----------")
        inf_output = run_inference(model, prep_image)
        #post_proc_results(inf_output)
        warmup_end_time = time.time()
        print("\033[96m" + "Time measured:" + f"\033[92m {warmup_end_time - warmup_start_time}" + "\033[0m")
    
    print("\n\n\n")

    try:
        num_iter = int(sys.argv[3])
        for x in range(num_iter):
            print(f"---------- ROUND {x + 1} ----------")
            start_time = time.time()
            inf_output = run_inference(model, prep_image)
            #post_proc_results(inf_output)
            end_time = time.time()
            total_inf_time = end_time - start_time
            print("\033[96m" + "Time measured:" + f"\033[92m {end_time - start_time}" + "\033[0m")

            check_slowdown(total_inf_time)

    except Exception as e:
        x = 0
        while True:
            print(f"---------- ROUND {x + 1} ----------")
            start_time = time.time()
            inf_output = run_inference(model, prep_image)
            #post_proc_results(inf_output)
            end_time = time.time()
            print("\033[96m" + "\nTime measured:" + f"\033[92m {end_time - start_time}" + "\033[0m")
            x = x + 1

            check_slowdown(total_inf_time)

    print("\n")
