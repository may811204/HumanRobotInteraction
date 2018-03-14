import numpy as np
import cv2
from prediction_server import model_generate
from data_pre_processor import preproces_data

# we just need front face classifier, eye is not needed
front_face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
IMAGE_SIZE = 48 # 48 x 48 image needed for CNN

def get_face(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  faces = front_face_classifier.detectMultiScale(
      image,
      scaleFactor = 1.3,
      minNeighbors = 5
  )
  # no face detected
  if not len(faces) > 0:
    return None
  # there will be many rectangles, find which has the maximum area
  # each rectangle has x, y, width, height
  max_area_face = faces[0]
  x_index = 0
  y_index = 1
  width_index = 2
  height_index = 3
  for face in faces:
    # check width * height = area greater than current face
    if face[width_index] * face[height_index] > max_area_face[width_index] * max_area_face[height_index]:
      max_area_face = face
  # chop image to face
  face = max_area_face
  image = image[face[y_index]:(face[y_index] + face[width_index]), face[x_index]:(face[x_index] + face[height_index])]
  # return image
  # Resize image to network size
  try:
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_CUBIC)
	# pre process the data for live frame (ZCA, GCN)
    image = preproces_data(image)
    #print(image.shape)
  except Exception:
    print("resize failed")
    return None
  return image
  
model = model_generate((48, 48, 1))
model.load_weights('best_model.6562.hdf5')

cap = cv2.VideoCapture(0)
emotion_map = {0: "Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = get_face(frame)
    # Display the resulting frame
    if face is not None:
       result = model.predict(face.reshape([-1, 48, 48, 1]))
	   #print result
       emotion = np.argmax(result)
       #print(emotion_map[emotion])
       cv2.putText(frame, emotion_map[emotion], (10, 50), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 1);
       cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()