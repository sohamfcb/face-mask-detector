from keras.models import load_model
import cv2

haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def detect_face(img):
    coods=haar.detectMultiScale(img)
    return coods

def detect_mask_from_image(img):
    y_pred=model.predict(img.reshape(1,224,224,3))
    y_pred=[0 if x < 0.5 else 1 for x in y_pred]
    final_pred=y_pred[0]
    return pred_dict[final_pred]

def face_mask_detector(img):
    y_pred=model.predict(img.reshape(1,224,224,3))
    y_pred=[0 if x < 0.5 else 1 for x in y_pred]
    return y_pred

def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)
    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)


if __name__=="__main__":

    model=load_model('model2')


    pred_dict={0:'Mask detected',1:'No mask detected'}
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, (224, 224))
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]

        y_pred = face_mask_detector(frame)

        if y_pred[0] == 0:
            draw_label(frame, 'Mask Detected', (30, 30), (0, 255, 0))
        else:
            draw_label(frame, 'No Mask', (30, 30), (0, 0, 255))

        cv2.imshow('window', frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cv2.destroyAllWindows()