
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
BIN_OUTPUT_PATH = "../bin_outputs/sar_preprocessed_input_data.bin"
context = None
trt_engine = None

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

def load_trt_engine(trt_path):
    try:
        print("\033[96m" + "Provided Engine filepath:" + "\033[0m \033[1m" + "{} \033[0m".format(trt_path))
        with open(trt_path, "rb") as f:
            print("\033[96m" + "Engine: " + "\033[92m" + "loaded" + "\033[0m \n")

            load_trt_engine_result = rad_log_start(30, b'load_trt_engine', b'load provided TRT engine filepath', str(trt_path).encode())
            print("\033[94mrad_logger:\033[0m TRT engine loading result logged for event id -> ", load_trt_engine_result)

            return RUNTIME.deserialize_cuda_engine(f.read())
    except Exception as e:
        print("\n" + "\033[91m" + f"Error loading engine from path {trt_path}:\n {e}" + "\033[0m \n")
        
        load_trt_engine_result = rad_log_start(31, b'load_trt_engine', b'load provided TRT engine filepath error!', str(e).encode())
        print("\033[94mrad_logger:\033[0m TRT engine loading result logged for event id -> ", load_trt_engine_result)

        exit()

def run_inference(engine, prep_image):
    try:
        print("\033[96m" + "Loading inference data" + "\033[0m")

        context = engine.create_execution_context()
        #print("\t- Created engine context")

        inf_data = prep_image.reshape((1,256,256,3))
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

        run_inference_result = rad_log_start(40, b'run_inference', b'inference run', str(outputs[0].host).encode())
        print("\033[94mrad_logger:\033[0m run inference result logged for event id -> ", run_inference_result)
        
        #print("outputs[0] -> ", outputs[0])
        #print("outputs[0].host -> ", outputs[0].host)
        return outputs[0].host
    except Exception as e:
        print("\n" + "\033[91m" + f"Error during inference stage: \n{e}" + "\033[0m \n")
        
        run_inference_result = rad_log_start(41, b'run_inference', b'error during inference run!', str(e).encode())
        print("\033[94mrad_logger:\033[0m run inference result logged for event id -> ", run_inference_result)

        exit()

def allocate_buffers(engine):
    inputs, outputs = [], []
    bindings = [None] * engine.num_bindings
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
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
    
    event_data = [inputs, outputs, bindings, stream]
    allocate_buffers_result = rad_log_start(10, b'allocate_buffers', b'allocate needed buffers for provided TRT engine', str(event_data).encode())
    print("\033[94mrad_logger:\033[0m buffer allocation result logged for event id -> ", allocate_buffers_result)

    return inputs, outputs, bindings, stream

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
    cleanup_result = rad_log_start(60, b'cleanup', b'cleanup runtime', b'none')
    print("\033[94mrad_logger:\033[0m cleanup results logged for event id -> ", cleanup_result)

# Check if inference time above threshold
def check_slowdown(total_inf_time):
    if total_inf_time > SLOW_THRESHOLD:
        # log time slowdown
        log_slow_time = rad_log_start(80, b'Inference time below threshold', b'inference slowdown', str(total_inf_time).encode())
        print("\033[91mrad_logger:\033[0m inference slowdown logged for event id -> ", log_slow_time)


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
            total_inf_time = end_time - start_time
            print("\033[96m" + "Time measured:" + f"\033[92m {end_time - start_time}" + "\033[0m")

            check_slowdown(total_inf_time)

    except Exception as e:
        x = 0
        while True:
            print(f"---------- ROUND {x + 1} ----------")
            start_time = time.time()
            inf_output = run_inference(trt_engine, prep_image)
            post_proc_results(inf_output)
            end_time = time.time()
            print("\033[96m" + "\nTime measured:" + f"\033[92m {end_time - start_time}" + "\033[0m")
            x = x + 1

            check_slowdown(total_inf_time)

    cleanup()

    print("\n")
