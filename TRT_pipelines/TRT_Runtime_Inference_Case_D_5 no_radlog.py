import tensorrt
import cv2
import numpy as np
import sys
import os

import pycuda.driver as cuda
import pycuda.autoinit

import time
import ctypes

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem


print("\n" + "\033[96m" + "TensorRT version:" + "\033[0m \033[1m" + "{} \033[0m".format(tensorrt.__version__))

librad = ctypes.CDLL('/usr/local/lib/librad_logger.so')
print("\n---------------------------")
print("| Loaded librad_logger.so |")
print("---------------------------\n")

TRT_LOGGER = tensorrt.Logger()
RUNTIME = tensorrt.Runtime(TRT_LOGGER)
BIN_OUTPUT_PATH = "../bin_outputs/eurosat_preprocessed_input_data.bin"
context = None
trt_engine = None

rad_log_start = librad.log_start

# Hard-coded slowdown time threshold
SLOW_THRESHOLD = 0.6

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

        image = image.astype(np.float32) / 255.0
        print("\t- Scaled to: 0, 1")
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        stddev = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (image - mean) / stddev
        print("\t- Generated normal of image")

        # Save as raw binary data (float32)
        norm_img.astype(np.float32).tofile(BIN_OUTPUT_PATH)
        print("\033[96m" + f"Preprocessed image saved to:" + "\033[0m \033[92m" + f"{BIN_OUTPUT_PATH}" + "\033[0m \n")
        
        #preprocess_data__log_result = rad_log_start(20, b'preprocess_data FINISH', b'preprocessing data', str(norm_img).encode())
        #print("\033[94mrad_logger:\033[0m Data preprocessing finished (event {})".format(preprocess_data__log_result))
        return norm_img

    except Exception as e:
        preprocess_data__log_result = rad_log_start(20, b'preprocess_data ERROR', b'preprocessing data', b'none')
        print("\033[91mrad_logger:\033[0m Data preprocessing error (event {})".format(preprocess_data__log_result))
        
        print("\033[91m" + f"Error during preprocessing: {e} \033[0m\n")
        sys.exit(1)

def load_trt_engine(trt_path):
    try:
        print("\033[96m" + "Provided Engine filepath:" + "\033[0m \033[1m" + "{} \033[0m".format(trt_path))
        with open(trt_path, "rb") as f:
            print("\033[96m" + "Engine: " + "\033[92m" + "loaded" + "\033[0m \n")
            return RUNTIME.deserialize_cuda_engine(f.read())
    except Exception as e:
        print("\n" + "\033[91m" + f"Error loading engine from path {trt_path}:\n {e}" + "\033[0m \n")
        exit()

def run_inference(engine, prep_image):
    try:
        print("\033[96m" + "Loading inference data" + "\033[0m")

        context = engine.create_execution_context()
        #print("\t- Created engine context")

        inf_data = prep_image.reshape((1,64,64,3))
        #print("\t- Pre-processed image reshapen")
        #print(f"\t- Shape of inf_data: {inf_data.shape}")

        inputs, outputs, bindings, stream = allocate_buffers(engine)
        #print("\t- Allocated buffers for TRT Engine")

        np.copyto(inputs[0].host, inf_data.ravel())
        for data_input in inputs:
            cuda.memcpy_htod_async(data_input.device, data_input.host, stream)
        #print("\t- Input data transferred to device")

        stream.synchronize()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            context.set_tensor_address(tensor_name, bindings[i])
        #print("\t- Tensor addresses set")

        print("\033[96m" + "Starting inference..." + "\033[0m")

        context.execute_async_v3(stream_handle=stream.handle)
        print("\033[96m" + "Inference ended" + "\033[0m")

        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
        #print("\033[96m" + "Predictions:" + "\033[92m" + " transferred back"  + "\033[0m \n")

        stream.synchronize()

        return outputs[0].host
    except Exception as e:
        print("\n" + "\033[91m" + f"Error during inference stage: \n{e}" + "\033[0m \n")
        exit()

def allocate_buffers(engine):
    inputs, outputs = [], []
    bindings = [None] * engine.num_io_tensors
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        binding = engine[i]
        size = tensorrt.volume(engine.get_tensor_shape(binding))
        shape = engine.get_tensor_shape(binding)

        host_mem = cuda.pagelocked_empty(size, dtype=np.float32)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings[i] = int(device_mem)
        if engine.get_tensor_mode(binding) == tensorrt.TensorIOMode.OUTPUT:
            outputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            inputs.append(HostDeviceMem(host_mem, device_mem))
    
    #event_data = [inputs, outputs, bindings, stream]
    #allocate_buffers__log_result = rad_log_start(10, b'allocate_buffers', b'allocate buffers FINISH', str(event_data).encode())
    #print("\033[94mrad_logger:\033[0m TRT engine buffer allocation finished (event {})".format(allocate_buffers__log_result))

    return inputs, outputs, bindings, stream

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
        
        #post_proc_category__log_result = rad_log_start(30, b'post_proc_category', b'Predicted category FINISH', str(PREDICTED_CLASS_NAME).encode())
        #print("\033[94mrad_logger:\033[0m Image class prediction finished (event {})".format(post_proc_category__log_result))

        #post_proc_probability__log_result = rad_log_start(31, b'post_proc_category', b'Predicted category FINISH', str(predicted_probability).encode())
        #print("\033[94mrad_logger:\033[0m Image prediction probability finished (event {})".format(post_proc_probability__log_result))

    except Exception as e:
        post_proc_category__log_result = rad_log_start(30, b'post_proc_category', b'Predicted category ERROR', b'none')
        print("\033[91mrad_logger:\033[0m Image class prediction finished (event {})".format(post_proc_category__log_result))

        post_proc_probability__log_result = rad_log_start(31, b'post_proc_category', b'Predicted category ERROR', b'none')
        print("\033[91mrad_logger:\033[0m Image prediction probability finished (event {})".format(post_proc_probability__log_result))

        print("\033[91m" + f"Error during post-processing results:\n {e}" + "\033[0m \n")
        exit()

# Check if inference time above threshold
def check_slowdown(total_inf_time):
    if total_inf_time > SLOW_THRESHOLD:
        # log time slowdown
        log_slow_time = rad_log_start(80, b'Inference time below threshold', b'inference slowdown', str(total_inf_time).encode())
        print("\033[91mrad_logger:\033[0m inference slowdown logged for event id -> ", log_slow_time)

def cleanup():
    global context, trt_engine, RUNTIME, TRT_LOGGER
    if context:
        del context
    if trt_engine:
        del trt_engine
    if RUNTIME:
        del RUNTIME
    if TRT_LOGGER:
        del TRT_LOGGER

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: <tensorrt_engine_path> <input_image_path> <optional_number_of_iterations>")
        sys.exit(1)

    if len(sys.argv) > 4:
        print("Usage: <tensorrt_engine_path> <input_image_path> <optional_number_of_iterations>")
        sys.exit(1)

    tensorrt_engine_path = sys.argv[1]
    input_image_path = sys.argv[2]
    trt_engine = load_trt_engine(tensorrt_engine_path)
    prep_image = preprocess_data(input_image_path)

    for i in range(5):
        warmup_start_time = time.time()
        print(f"---------- WARMUP {i + 1} ----------")
        inf_output = run_inference(trt_engine, prep_image)
        post_proc_results(inf_output)
        warmup_end_time = time.time()
        print("\033[96m" + "Time measured:" + f"\033[92m {warmup_end_time - warmup_start_time}" + "\033[0m")
    
    print("\n\n\n")

    try:
        num_iter = int(sys.argv[3])
        for x in range(num_iter):
            print(f"---------- ROUND {x + 1} ----------")
            start_time = time.time()
            inf_output = run_inference(trt_engine, prep_image)
            post_proc_results(inf_output)
            end_time = time.time()

            total_time = end_time - start_time
            print("\033[96m" + "Time measured:" + f"\033[92m {end_time - start_time}" + "\033[0m")

            #total_inf_time__log_result = rad_log_start(40, b'inference_time', b'Inference time calculation FINISH', str(total_time).encode())
            #print("\033[94mrad_logger:\033[0m Inference time finished (event {})".format(total_inf_time__log_result))

            check_slowdown(total_time)
    except Exception as e:
        x = 0
        while True:
            print(f"---------- ROUND {x + 1} ----------")
            start_time = time.time()
            inf_output = run_inference(trt_engine, prep_image)
            post_proc_results(inf_output)
            end_time = time.time()
            total_time = end_time - start_time
            print("\033[96m" + "Time measured:" + f"\033[92m {end_time - start_time}" + "\033[0m")
            check_slowdown(total_time)
            x = x + 1

    cleanup()

    print("\n")
