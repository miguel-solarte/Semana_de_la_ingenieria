import cv2
from model_torch import model_pre
import time

def deteccion(path,model,preprocess,color_boxes):

    cap = cv2.VideoCapture(path)
    



    while cap.isOpened():

        #ini = time.time()

        _, image = cap.read()
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cv2.destroyAllWindows()
            break

        #image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #img = cv2.resize(image1, (500,500))
        boxes, labels, scores = model_pre(image, model, preprocess)
        k = 0

        for j,i in enumerate(labels):
            

            if i == 1:
                k += 1

                start_point = boxes[j,:2].tolist()

                end_point =boxes[j,2:].tolist()

                image_boxes = cv2.rectangle(image, start_point, end_point, color_boxes, 2)
        #fin = time.time()
        try:
            cv2.putText(image_boxes,f'NUMERO DE PERSONAS: {k}',(50, 50),1,cv2.FONT_HERSHEY_COMPLEX, (0,0,0),2)
            cv2.imshow('video ITM', image_boxes)
        
        except:
            print('error')
        if cv2.waitKey(1) == ord('q'):
            break
    
    
    

    cap.release()
    cv2.destroyAllWindows()