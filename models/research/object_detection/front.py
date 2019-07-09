from tkinter import *
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image, ImageTk

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
root = Tk()
root.title("BLURING")

answer_label =Label(root, text ="---")
answer_label.grid(row =0, column =0)

label1 =Label(root, text ="Input file")
label1.grid(row =1, column =0)

num1_txtbx =Entry(root)
num1_txtbx.grid(row =1, column =1)

def vid():
    if(num1_txtbx.get() != ""):
        try:
            VIDEO_NAME = str(num1_txtbx.get())
            answer = VIDEO_NAME
            CWD_PATH = os.getcwd()
            PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
            PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
            PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)
            NUM_CLASSES = 7
            label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)

            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                sess = tf.Session(graph=detection_graph)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            video = cv2.VideoCapture(PATH_TO_VIDEO)
            count=0
            while(video.isOpened()):
                ret, frame = video.read()
                #cv2.imwrite("frame%d.jpg" % count, frame)
                frame_expanded = np.expand_dims(frame, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=0.60)
                from PIL import Image, ImageFilter
                #im = Image.open('frame%d.jpg'%count)
                #width, height = im.size
                #print(width, height)
                width = video.get(3)
                height = video.get(4)
                width=int(width)
                height=int(height)

                box = np.squeeze(boxes)
                for i in range(len(boxes)):
                    ymin = (int(box[i,0]*height))
                    xmin = (int(box[i,1]*width))
                    ymax = (int(box[i,2]*height))
                    xmax = (int(box[i,3]*width))
                    print(ymin,xmin,ymax,xmax)
                print(boxes)

                cropped_image = frame[ymin:ymax,xmin:xmax].copy()
                cropped_image = Image.fromarray(cropped_image)
                blurred_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=20))
                im.paste(blurred_image,(xmin,ymin,xmax,ymax))
                

                count+=1
                b, g, r = im.split()
                im = Image.merge("RGB", (r, g, b))
                im = np.array(im)
                cv2.imshow('Object detector', im)
                #im.show()
                video.release()
            answer_label.configure(text =answer)
            status_label.configure(text ="successfully blurred")
            
        except:
            status_label.configure(text ="invalid input, check your input types")
    else:
        status_label.configure(text ="fill in all the required fields")

            

def addF():
    if(num1_txtbx.get() != ""):
        try:
            IMAGE_NAME =str(num1_txtbx.get())
            answer = IMAGE_NAME
            CWD_PATH = os.getcwd()
            PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
            PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
            PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
            NUM_CLASSES = 7
            label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)

            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                sess = tf.Session(graph=detection_graph)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image = cv2.imread(PATH_TO_IMAGE)
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.60)

            from PIL import Image, ImageFilter
            im = Image.open(answer)
            width, height = im.size

            box = np.squeeze(boxes)
            for i in range(len(boxes)):
                ymin = (int(box[i,0]*height))
                xmin = (int(box[i,1]*width))
                ymax = (int(box[i,2]*height))
                xmax = (int(box[i,3]*width))

            cropped_image = im.crop((xmin,ymin,xmax,ymax))
            blurred_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=20))
            im.paste(blurred_image,(xmin,ymin,xmax,ymax))

            
            im.show()
            
            answer_label.configure(text =answer)
            status_label.configure(text ="successfully blurred")
            
        except:
            status_label.configure(text ="invalid input, check your input types")
    else:
        status_label.configure(text ="fill in all the required fields")
        

def fonk():
    if(num1_txtbx.get() == ""):
        IMAGE_NAME =str(num1_txtbx.get())
        answer = IMAGE_NAME
        CWD_PATH = os.getcwd()
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
        PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
        PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
        NUM_CLASSES = 7
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        video = cv2.VideoCapture(0)
        ret = video.set(3,1280)
        ret = video.set(4,720)

        count=0

        while(True):
            ret, frame = video.read()
            cv2.imwrite("frame%d.jpg" % count, frame)
            frame_expanded = np.expand_dims(frame, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.85)
            from PIL import Image, ImageFilter
            im = Image.open('frame%d.jpg'%count)
            width, height = im.size
            print(width, height)

            box = np.squeeze(boxes)

            for i in range(len(boxes)):
                ymin = (int(box[i,0]*height))
                xmin = (int(box[i,1]*width))
                ymax = (int(box[i,2]*height))
                xmax = (int(box[i,3]*width))
                print(ymin,xmin,ymax,xmax)
            print(boxes)


            cropped_image = im.crop((xmin,ymin,xmax,ymax))
            blurred_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=20))
            im.paste(blurred_image,(xmin,ymin,xmax,ymax))
            count += 1
            im.show()


            
            #cv2.imshow('Object detector', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
        status_label.configure(text ="successfully detection")
    
    else:
        status_label.configure(text ="don't fill in all the required fields")

                

calculate_button =Button(root, text="Blur Image", command= addF)
calculate_button.grid(row =3, column =0)

web_button =Button(root, text="Webcam Detection", command= fonk)
web_button.grid(row =4, column =0)

web_button =Button(root, text="Blur Video", command= vid)
web_button.grid(row =5, column =0)

status_label =Label(root, height =10, width =80, bg ="white", fg ="#00FF00", text ="---", wraplength =150)
status_label.grid(row =6, column =0, columnspan =2)

root.mainloop()
