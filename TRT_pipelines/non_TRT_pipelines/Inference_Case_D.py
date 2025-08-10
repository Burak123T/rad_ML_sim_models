
import cv2
import numpy as np
import tensorflow as tf
import keras

import sys
import os

import time
import ctypes



librad = ctypes.CDLL('/usr/local/lib/librad_logger.so')
print("\n---------------------------")
print("| Loaded librad_logger.so |")
print("---------------------------\n")

BIN_OUTPUT_PATH = "../../bin_outputs/eurosat_preprocessed_input_data_keras.bin"

rad_log_start = librad.log_start

# int event_id, const char *event_name, const char *event_type, const char *event_data
arg_types = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
rad_log_start.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]

CATEGORIES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

def preprocess_data(image_path):
    try:
        print("\033[96m" + "Reading input for preprocessing:" + "\033[0m \033[1m" + "{} \033[0m".format(image_path))

        # Load provided image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not open or read image file: {image_path}")
        else:
            print("\t- loaded image: {}".format(image_path))

        resized_image_array = cv2.resize(image, (64, 64))
        _img_array = np.array(resized_image_array)
        
        _image = _img_array.astype(np.float32) / 255.0
        print("\t- Scaled to: 0, 1")
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        stddev = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (_image - mean) / stddev
        print("\t- Generated normal of image")

        # Needed due to size mismatch
        norm_img_expand = np.expand_dims(norm_img, axis=0)

        # Save as raw binary data (float32)
        norm_img_expand.astype(np.float32).tofile(BIN_OUTPUT_PATH)
        print("\033[96m" + f"Preprocessed image saved to:" + "\033[0m \033[92m" + f"{BIN_OUTPUT_PATH}" + "\033[0m \n")
        
        preprocess_data__log_result = rad_log_start(20, b'preprocess_data FINISH', b'preprocessing data', str(norm_img).encode())
        print("\033[94mrad_logger:\033[0m Data preprocessing finished (event {})".format(preprocess_data__log_result))
        return norm_img_expand

    except Exception as e:
        preprocess_data__log_result = rad_log_start(20, b'preprocess_data ERROR', b'preprocessing data', b'none')
        print("\033[91mrad_logger:\033[0m Data preprocessing error (event {})".format(preprocess_data__log_result))
        
        print("\033[91m" + f"Error during preprocessing: {e} \033[0m\n")
        sys.exit(1)

def run_inference(model, prep_image):
    try:
        print("\033[96m" + "Predicting..." + "\033[0m")
        model.predict(prep_image)
        #model.evaluate(prep_image)
        
    except Exception as e:
        print("\n" + "\033[91m" + f"Error during inference stage: \n{e}" + "\033[0m \n")
        exit()

def post_proc_results(pred_output):

    print("\033[96m" + "Running post-processing" + "\033[0m")

    try:
        predictions = pred_output.flatten()
        #print("\t- Inference output flattened")

        PREDICTED_CLASS_INDEX = np.argmax(predictions)
        PREDICTED_CLASS_NAME = CATEGORIES[PREDICTED_CLASS_INDEX]
        #print("\t- Defined PREDICTED_CLASS_INDEX and PREDICTED_CLASS_NAME\n")

        predicted_probability = predictions[PREDICTED_CLASS_INDEX]

        print("\033[96m" + "Predicted category:" + f"\033[92m {PREDICTED_CLASS_NAME}" + "\033[0m")
        print("\033[96m" + f"Probability:" + f"\033[92m {(predicted_probability * 100):.3f}%" + "\033[0m")
        
        post_proc_category__log_result = rad_log_start(30, b'post_proc_category', b'Predicted category FINISH', str(PREDICTED_CLASS_NAME).encode())
        print("\033[94mrad_logger:\033[0m Image class prediction finished (event {})".format(post_proc_category__log_result))

        post_proc_probability__log_result = rad_log_start(31, b'post_proc_category', b'Predicted category FINISH', str(predicted_probability).encode())
        print("\033[94mrad_logger:\033[0m Image prediction probability finished (event {})".format(post_proc_probability__log_result))

    except Exception as e:
        post_proc_category__log_result = rad_log_start(30, b'post_proc_category', b'Predicted category ERROR', b'none')
        print("\033[91mrad_logger:\033[0m Image class prediction finished (event {})".format(post_proc_category__log_result))

        post_proc_probability__log_result = rad_log_start(31, b'post_proc_category', b'Predicted category ERROR', b'none')
        print("\033[91mrad_logger:\033[0m Image prediction probability finished (event {})".format(post_proc_probability__log_result))

        print("\033[91m" + f"Error during post-processing results:\n {e}" + "\033[0m \n")
        exit()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: <keras_model> <input_image_path> <optional_number_of_iterations>")
        sys.exit(1)

    if len(sys.argv) > 4:
        print("Usage: <keras_model> <input_image_path> <optional_number_of_iterations>")
        sys.exit(1)

    keras_model_path = sys.argv[1]
    input_image_path = sys.argv[2]
    model = keras.models.load_model(keras_model_path)
    print(model.summary())
    prep_image = preprocess_data(input_image_path)

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

            total_time = end_time - start_time
            print("\033[96m" + "Time measured:" + f"\033[92m {end_time - start_time}" + "\033[0m")

            total_inf_time__log_result = rad_log_start(40, b'inference_time', b'Inference time calculation FINISH', str(total_time).encode())
            print("\033[94mrad_logger:\033[0m Inference time finished (event {})".format(total_inf_time__log_result))
    except Exception as e:
        x = 0
        while True:
            print(f"---------- ROUND {x + 1} ----------")
            start_time = time.time()
            inf_output = run_inference(model, prep_image)
            #post_proc_results(inf_output)
            end_time = time.time()
            print("\033[96m" + "Time measured:" + f"\033[92m {end_time - start_time}" + "\033[0m")
            x = x + 1

    print("\n")
