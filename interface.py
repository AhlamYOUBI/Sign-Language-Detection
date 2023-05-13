import tkinter as tk
from PIL import ImageTk, Image
import cv2
import mediapipe as mp
import pickle
import numpy as np
from tkinter import messagebox
import time


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
                9:'J', 10:'k', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q',
                17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}
data_aux = []
x_ = []
y_ = []

class SignLanguageGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Détection des signes de la langue des signes")
        self.window.geometry("900x500")
        self.window.resizable(False, False)

        # Ajout d'un bouton de capture d'image
        self.capture_button = tk.Button(window,height=1, width=10, text="Capture", command=self.capture)
        self.capture_button.pack()
        
        # Ajout d'une zone d'affichage d'image
        self.image_label = tk.Label(window)
        self.image_label.pack()

        #Ajout d'une zone d'affichage d'image
        self.canvas_width = 400
        self.canvas_height = 400
        self.resultat_canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height)
        self.resultat_canvas.pack(side=tk.RIGHT, padx=10, pady=10)
        # Ajout d'une zone d'affichage d'image
        self.canvas_width = 400
        self.canvas_height = 400
        self.image_canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height)
        self.image_canvas.pack(side=tk.LEFT, padx=10, pady=10)


    

    def capture(self):
        print("capture d'image")

        # Ouverture de la webcam
        capture = cv2.VideoCapture(0)


        # Détection des mains à partir d'images
        mp_hands = mp.solutions.hands  
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.3)

        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']

        labels_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
                        9:'J', 10:'k', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q',
                        17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}


        # Affichage du message "Veuillez positionner devant la caméra" dans la caméra"
        ret, frame = capture.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Veuillez positionner devant la caméra", (50, 50), font, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)

        # Affichage du compte à rebours "La capture sera effectuée dans 3s, 2s, 1s"
        for i in range(10, 0, -1):
            ret, frame = capture.read()
            cv2.putText(frame, f"La capture sera effectuee dans {i} seconde(s)...", (50, 50), font, 1, (0, 0, 255), 2)
            cv2.imshow('frame', frame)
            cv2.waitKey(1000)

        # Capture de l'image
        ret, frame = capture.read()
        ##
        

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks :
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Prédiction de la lettre
            prediction = model.predict([np.asarray(data_aux)])

            predicted_char = labels_dict[int(prediction[0])]
            print(predicted_char)





        ##
        # Enregistrement de l'image capturée
        cv2.imwrite("capture.png", frame)

        # Libération des ressources de la webcam
        capture.release()

        hands.close()

        # Affichage de l'image capturée dans la zone d'affichage
        image = Image.open("capture.png")
        image = image.resize((self.canvas_width, self.canvas_height), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(image)
        # Afficher la capture dans le Canvas
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        # Affichage du résultat dans le canvas droit
        self.resultat_canvas.delete("all")
        self.resultat_canvas.create_text(self.canvas_width//2, self.canvas_height//2, text=f"Le résultat est la lettre: {predicted_char}", font=('Arial', 24), fill='black')



        

    
        
    



root = tk.Tk()
gui = SignLanguageGUI(root)
root.mainloop()