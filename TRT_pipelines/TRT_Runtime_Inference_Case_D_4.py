import tensorrt
import cv2
import numpy as np
import sys
import os

import pycuda.driver as cuda
import pycuda.autoinit

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem


print("\n" + "\033[96m" + "TensorRT version:" + "\033[0m \033[1m" + "{} \033[0m".format(tensorrt.__version__))

TRT_LOGGER = tensorrt.Logger()
RUNTIME = tensorrt.Runtime(TRT_LOGGER)
BIN_OUTPUT_PATH = "eurosat_preprocessed_input_data.bin"
context = None
trt_engine = None 

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

        return norm_img

    except Exception as e:
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
        print("\t- Created engine context")
        
        inf_data = prep_image.reshape((1,64,64,3))
        print("\t- Pre-processed image reshapen")
        print(f"\t- Shape of inf_data: {inf_data.shape}")

        inputs, outputs, bindings, stream = allocate_buffers(engine)
        print("\t- Allocated buffers for TRT Engine")

        np.copyto(inputs[0].host, inf_data.ravel())
        for data_input in inputs:
            cuda.memcpy_htod_async(data_input.device, data_input.host, stream)
        print("\t- Input data transferred to device")

        stream.synchronize()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            context.set_tensor_address(tensor_name, bindings[i])
        print("\t- Tensor addresses set")

        print("\t- Starting inference...")

        context.execute_async_v3(stream_handle=stream.handle)
        print("\t- Inference ended")

        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
        print("\t- Predictions transferred back")

        stream.synchronize()

        return outputs[0].host
    except Exception as e:
        print("\n" + "\033[91m" + f"Error during inference stage: \n{e}" + "\033[0m \n")
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
            print("\t\tOutput Host:\n\t\t" + str(host_mem) + "\n\t\tOutput Device:\n\t\t" + str(device_mem))
        else:
            inputs.append(HostDeviceMem(host_mem, device_mem))
            print("\t\tInput Host:\n\t\t" + str(host_mem) + "\n\t\tInput Device:\n\t\t" + str(device_mem))
    
    
    
    return inputs, outputs, bindings, stream

def post_proc_results(pred_output):
    try:
        predictions = pred_output.flatten()

        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CATEGORIES[predicted_class_index]

        predicted_probability = predictions[0][predicted_class_index]

        print(f"Predicted category: {predicted_class_name}")
        print(f"Probability: {(predicted_probability * 100):.2f}%")

        # Optional: Print probabilities for all categories (for debugging or more detail)
        print("\nProbabilities for all categories:")
        for i, category_name in enumerate(CATEGORIES):
            probability = predictions[0][i]
            print(f"  {category_name}: {(probability * 100):.2f}%")
    except Exception as e:
        print("\033[91m" + f"Error during post-processing results:\n {e}" + "\033[0m \n")
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

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: <tensorrt_engine_path> <input_image_path>")
        sys.exit(1)

    tensorrt_engine_path = sys.argv[1]
    input_image_path = sys.argv[2]

    trt_engine = load_trt_engine(tensorrt_engine_path)
    prep_image = preprocess_data(input_image_path)
    run_inference(trt_engine, prep_image)
    cleanup()
    #post_proc_results(1)

