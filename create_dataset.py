import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands  
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1, min_tracking_confidence=0.1)

DATA_DIR = './data'

data = []
labels = []


for dir_ in os.listdir(DATA_DIR) :  
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)) :  #[:1]  
        data_array = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_array.append(x - min(x_))
                    data_array.append(y - min(y_))

            data.append(data_array)
            labels.append(dir_)
                   
        else:
            print('No hand detected in image', dir_ , img_path)


##################################################################################################
# Sauvegarder les données collectées (les coordonnees des points d'interet de la main)
# et les étiquettes associées (classes) dans un fichier binaire appelé "data.pickle". 
# Cela permet de stocker les données dans un format qui peut être facilement lu et chargé ultérieurement, 
# ce qui est utile pour la formation et l'évaluation de modèles d'apprentissage automatique.
##################################################################################################

f = open('data.pickle', 'wb')  #write binary
pickle.dump({'data' : data , 'labels' : labels}, f)
f.close()




