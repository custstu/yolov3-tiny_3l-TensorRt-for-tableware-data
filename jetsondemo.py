# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:44:03 2020

@author: chengfu.liao
"""

#TRT inference
import cv2
import serial
import sys
import multiprocessing
from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from PIL import ImageDraw
from data_processing import PreprocessYOLO, PostprocessYOLO,  ALL_CATEGORIES


# 1.open camera
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('1.Open camera successful.')
else:
    print('Error: Open camera failed.')
    exit(1)

# 2.open serial port
ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=0)
if ser.isOpen():
    print('2.Open serial port successful.')
else:
    print('Error: Open serial port failed.')
    exit(1)
ser.flushInput()


# put images into the queue
def write_queue(queue):
   
    while cap.isOpened() and ser.isOpen():
        ret, frame = cap.read()
        count = ser.inWaiting()
        if count != 0:
            recv = ser.read(count)
            if str(recv,'utf-8') == 'b':
                ret, frame = cap.read()
                if ret == 0:
                    print('Error: Take photo failed.')
                    cap.release()
                    ser.close()
                    break
                queue.put(frame)
                #imshow image
                '''
                ...
                ...
                '''
                time.sleep(0)
                ser.flushInput()



def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.
    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    #print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]



# read images from the queue
def read_queue(queue):
    # 3.load model
    # initialize
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    runtime = trt.Runtime(TRT_LOGGER)


    # create engine
    with open('model.bin', 'rb') as f:
        buf = f.read()
        engine = runtime.deserialize_cuda_engine(buf)

    # create buffer
    host_inputs  = []
    cuda_inputs  = []
    host_outputs = []
    cuda_outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        host_mem = cuda.pagelocked_empty(size, np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    context = engine.create_execution_context()

    batch_size = 1
    input_size = 416
    output_shapes_416 = [(batch_size, 54, 13, 13), (batch_size, 54, 26, 26), (batch_size, 54, 52, 52)]
    output_shapes_480 = [(batch_size, 54, 15, 15), (batch_size, 54, 30, 30), (batch_size, 54, 60, 60)]
    output_shapes_544 = [(batch_size, 54, 17, 17), (batch_size, 54, 34, 34), (batch_size, 54, 68, 68)]
    output_shapes_608 = [(batch_size, 54, 19, 19), (batch_size, 54, 38, 38), (batch_size, 54, 72, 72)]
    output_shapes_dic = {'416': output_shapes_416, '480': output_shapes_480, '544': output_shapes_544, '608': output_shapes_608}
    

    output_shapes = output_shapes_dic[str(input_size)]
    input_resolution_yolov3_HW = (input_size, input_size)
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)

    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
                    "yolo_anchors": [(4,7), (7,15), (13,25),   (25,42), (41,67), (75,94),   (91,162), (158,205), (250,332)],
                    "obj_threshold": 0.5, 
                    "nms_threshold": 0.35,
                    "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    print('3.Load model successful.')
    print('Everything is ready.')

    num = 0
    while cap.isOpened() and ser.isOpen():
        if queue.empty():
            continue
        frame = queue.get()
        images = []
        image_raw, image = preprocessor.process(frame)
        images.append(image)
        num = num + 1
        images_batch = np.concatenate(images, axis=0)
        inputs[0].host = images_batch
        #t1 = time.time()
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=batch_size)
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
        shape_orig_WH = image_raw.size
        boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH), 0)
        #t2 = time.time()
        #t_inf = t2 - t1
        #print("time consumption:",t_inf)
        print(boxes, scores, classes)
        images.clear()
        if np.all(scores == 0):
            ser.write("h".encode("utf-8"))
            print('exception.')
            continue	
        index = np.nonzero(classes)
        label = classes[index[0]]
        cv2.imwrite('tmp/'+str(num)+'.jpg', frame)
        if label == 0:
            ser.write("c".encode("utf-8"))
            print('plate front.')
        elif label == 1:
            ser.write("d".encode("utf-8"))
            print('plate back.')
        elif label == 2:
            ser.write("f".encode("utf-8"))
            print('bowl front.')
        elif label == 3:
            ser.write("e".encode("utf-8"))
            print('bowl back.')
        elif label == 4:
            ser.write("g".encode("utf-8"))
            print('glass cup side.')
        elif label == 5:
            ser.write("g".encode("utf-8"))
            print('glass cup back.')
        elif label == 6:
            ser.write("g".encode("utf-8"))
            print('glass cup front.')
        elif label == 7:
            ser.write("i".encode("utf-8"))
            print('teacup side.')
        elif label == 8:
            ser.write("j".encode("utf-8"))
            print('teacup back.')
        elif label == 9:
            ser.write("k".encode("utf-8"))
            print('teacup front.')
        elif label == 10:
            ser.write("g".encode("utf-8"))
            print('cup side.')
        elif label == 11:
            ser.write("g".encode("utf-8"))
            print('cup back.')
        elif label == 12:
            ser.write("g".encode("utf-8"))
            print('cup front.')
        else:
            ser.write("h".encode("utf-8"))
            print('exception.')


if __name__ == '__main__':
    # create queue
    queue = multiprocessing.Queue()
    # write processing
    pw = multiprocessing.Process(target=write_queue, args=(queue,))
    # read processing
    pr = multiprocessing.Process(target=read_queue, args=(queue,))
    pw.start()
    pr.start()
    pw.join()
