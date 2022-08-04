from kivymd.app import MDApp
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from keras.models import load_model
from keras.utils import img_to_array

import cv2
import numpy as np

print("why: ",MDApp)
emotion_model = load_model('emotion_detection_model_100epochs.h5')
age_model = load_model('age_model_50epochs.h5')
gender_model = load_model('gender_model_50epochs.h5')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']



class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # added


        ret, frame = self.capture.read()
        if ret:
            labels=[]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # added
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)  # added

            for (x, y, w, h) in faces:  # added
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    # Get image ready for prediction
                    roi = roi_gray.astype('float') / 255.0  # Scale
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)  # Expand dims to get it ready for prediction (1, 48, 48, 1)

                    preds = emotion_model.predict(roi)[0]  # Yields one hot encoded result for 7 classes
                    label = class_labels[preds.argmax()]  # Find the label
                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Gender
                    roi_color = frame[y:y + h, x:x + w]
                    roi_color = cv2.resize(roi_color, (200, 200), interpolation=cv2.INTER_AREA)
                    gender_predict = gender_model.predict(np.array(roi_color).reshape(-1, 200, 200, 3))
                    gender_predict = (gender_predict >= 0.5).astype(int)[:, 0]
                    gender_label = gender_labels[gender_predict[0]]
                    gender_label_position = (x, y + h + 50)  # 50 pixels below to move the label outside the face
                    cv2.putText(frame, gender_label, gender_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Age
                    age_predict = age_model.predict(np.array(roi_color).reshape(-1, 200, 200, 3))
                    age = round(age_predict[0, 0])
                    age_label_position = (x + h, y + h)
                    cv2.putText(frame, "Age=" + str(age), age_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)




            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture


class CamApp(MDApp):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        return self.my_camera

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()


if __name__ == '__main__':
    CamApp().run()