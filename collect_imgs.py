import os
import cv2


# créer un dossier "data" à l'emplacement courant du script pour stocker les images prétraitées
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# le nombre de classes à collecter
number_of_classes = 26
# la taille de chaque classe (Nb d'images)
dataset_size = 150


######################################################################################################
# Ce code parcourt les dossiers contenant les images de chaque classe de la base de données ASL. 
# Il crée un nouveau dossier nommé "data" pour stocker les images redimensionnées.
# Ensuite, il lit chaque image dans chaque dossier de classe, 
# redimensionne l'image en une dimension de 224x224 pixels en utilisant la fonction cv2.resize (), 
# puis enregistre l'image redimensionnée dans un sous-dossier approprié de "data". 
# Le code s'arrête d'enregistrer les images après avoir atteint la taille du dataset spécifiée.
#######################################################################################################

for j in range(number_of_classes):
    class_dir = os.path.join(r'C:/Users/Admin/Desktop/ML ASL/Dataset_Kaggle/asl_alphabet_train', str(j))
    if not os.path.exists(class_dir):
        print(f"Class {j} is missing, skipping.")
        continue

    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    counter = 0
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}, skipping.")
            continue

        img = cv2.resize(img, (200, 200))
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), img)
        counter += 1
        if counter >= dataset_size:
            break

cv2.destroyAllWindows()
