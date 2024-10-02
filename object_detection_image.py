import cv2

#--------READ DNN MODEL--------
#MODEL ARQUITECTURE

propotxt = "model/MobileNetSSD_deploy.prototxt.txt"
# Weight
model = "model/MobileNetSSD_deploy.caffemodel"
#class labels
classes = {0:"background", 1:"aerplane", 2:"bicycle",
           3:"bird", 4:"boat",
           5:"bottle", 6:"bus",
           7:"car", 8:"cat",
           9:"chair", 10:"cow",
           11:"diningtable", 12:"dog",
           13:"horse", 14:"motorbike",
           15:"person", 16:"pottendplant",
           17:"sheep", 18:"sofa",
           19:"train", 20:"tvmonitor"}

# Load the model
net = cv2.dnn.readNetFromCaffe(propotxt, model)

#---------- READ THE IMAGE AND PREPROCESSING --------
image = cv2.imread("ImagesVideos/image1")
# height, width,_ = image.shape
image_resized = cv2.resize(image, (300, 300)) #Recomended values 

#Create a blob
blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5)) # THIS VALUES ARE FROM THE DOCUMENTATION OF "OPENCV" DONT MODIFICATED


# -------- DETECTION AND PREDICTIONS WHIT AI -------
net.setInput(blob)
for detection in detections[0][0]:
    if detection[2] > 0.45: #BEST VALUE FOR AI CAN RECOGNIZE
        label = classes[detection[1]]
        box = detection[3:7] * [width, height, width, height]
        x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(image, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 0, 0), 2)
        cv2.putText(image, label, (x_start, y_start - 5), 1, 1.2, (255, 0, 0), 2)


cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
