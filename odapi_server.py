import tensorflow as tf
from object_detection.utils import label_map_util
import numpy as np
import os
import time
import matplotlib.image as mpimg
from imgaug import augmenters as iaa

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/zlong/data/download/download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'bug/data/label_map.pbtxt'

NUM_CLASSES = 3

MAX_WIDTH = 1024


os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 定义TensorFlow配置
config = tf.ConfigProto(allow_soft_placement=True)
# 配置GPU内存分配方式，按需增长，很关键
config.gpu_options.allow_growth = True
# 配置可使用的显存比例
config.gpu_options.per_process_gpu_memory_fraction = 0.4

# 加载模型
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# 加载类型
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 将图片加载到numpy数组中
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def detect(image_path):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config= config) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            start_time = time.time()
            image = mpimg.imread(image_path)

            print('===Read Spend:', time.time() - start_time)
            start_time = time.time()


            # if image.shape[1] > MAX_WIDTH:
            #     aug = iaa.Scale({"width": MAX_WIDTH, "height": "keep-aspect-ratio"})
            #     image = aug.augment_images([image])[0]
            #
            #     print('===Scale Spend:', time.time() - start_time)
            #     start_time = time.time()

            image_np_expanded = np.expand_dims(image, axis=0)
            print('===Expand Spend:', time.time() - start_time)
            start_time = time.time()

            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            print('===Detection Spend:', time.time() - start_time)
            start_time = time.time()

            result = []
            for score in scores[0]:
                if score > 0.5:
                    index = scores[0].tolist().index(score)
                    cls = int(classes[0][index]);
                    box = boxes[0][index]
                    print(cls, score)
                    result.append({'cls': cls, 'box': box.tolist(), 'score': score})
            # sess.reset('')
            sess.close()

    return result;
