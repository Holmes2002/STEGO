import pycuda.driver as cuda
import pycuda.autoinit
from scipy.special import softmax
import numpy as np
import sys
import tensorrt as trt
import time
import os
import cv2 
from PIL import Image
TRT_LOGGER = trt.Logger()

def padding(img):
    w,h = img.size 
    if w<h:
        scale = w/h
        new_size = (int(224*scale),224)
        img = img.resize(new_size, Image.NEAREST)
        paddimg_w  = 224- int(224*scale)
        result = Image.new(img.mode, (224, 224), 0)
        result.paste(img, (int(paddimg_w//2), 0))
    else :
        scale = h/w
        new_size = (224,int(224*scale))
        img = img.resize(new_size, Image.NEAREST)
        padding_h  = 224- int(224*scale)
        result = Image.new(img.mode, (224, 224), 0)
        result.paste(img, (0, int(padding_h//2)))
    return result
def preprocess(img):

    img = padding(img)
    mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    img = (np.array(img).astype('float32') / float(255.0) - mean) / stddev
    # Switch from HWC to to CHW order
    return   np.moveaxis(img, 2, 0)
def postprocess(output_buffer):
        output_buffer = np.transpose(output_buffer, (1,2,0))
        output_buffer = cv2.resize(output_buffer, (224,224))
        output_buffer = softmax(output_buffer, axis=2)
        output_buffer = np.argmax(output_buffer, axis = 2)
        return output_buffer
def load_engine(engine_file_path = 'fp16.trt' ):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
def infer(engine, img_path, file):
    start = time.time()
    img = Image.open(img_path) 
    input_image = preprocess(img)
    image_width = 224
    image_height = 224
    with engine.create_execution_context() as context:

        # Set input shape based on image dimensions for inference
        context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))

        # Allocate host and device buffers
        print('Size ')

        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):

                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()
        output_buffer = np.reshape(output_buffer, (9,28,28))
        output_buffer = postprocess(output_buffer)
        # output_buffer = postprocess(output_buffer)
        end = time.time()
        cv2.imwrite('results/'+ file.replace('jpg', 'png'),output_buffer)
        print(f'FPS {1/(end-start)}')
if __name__ == '__main__':
    engine = load_engine( 'fp16.trt' )
    path_images = 'dataset/images'
    for file in os.listdir(path_images):
        img_path = os.path.join(path_images, file)
        infer( engine, img_path, file)

