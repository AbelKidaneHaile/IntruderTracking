# imports
import tensorflow as tf
import cv2
from sklearn.preprocessing import normalize
import numpy as np
import os 
from retinaface import RetinaFace

#--------------------------------------------------------------------------------------------------------------
def list_files_in_directory(directory_path):
    try:
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        return files
    except OSError:
        print(f"Error listing files in {directory_path}")
        return []

#--------------------------------------------------------------------------------------------------------------

def generate_query_embeddings(query_folder_path, face_model):
    query_embeddings = []
    files_list = list_files_in_directory(query_folder_path)
    for query_path in files_list: 
        query_embedding = generate_embeddings(os.path.join(query_folder_path, query_path), face_model)
        query_embeddings.append(query_embedding)

    return query_embeddings

#--------------------------------------------------------------------------------------------------------------
def generate_embeddings_img(img, face_model):
    nimg1 = []
    img = cv2.resize(img, (112, 112))
    nimg1.append(img)
    nimg1 = (np.array(nimg1) - 127.5)* 0.0078125
    embedding1 = face_model(nimg1).numpy()
    embedding1 = normalize(embedding1)

    return embedding1
#--------------------------------------------------------------------------------------------------------------
def generate_embeddings(image_path, face_model):
    nimg1 = []
    img =cv2.imread(image_path)
    faces = RetinaFace.extract_faces(img, align = True)
    for face in faces:
        img = face
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        nimg1.append(img)
        nimg1 = (np.array(nimg1) - 127.5)* 0.0078125
        embedding1 = face_model(nimg1).numpy()
        embedding1 = normalize(embedding1)
        return embedding1
    return None
 
#--------------------------------------------------------------------------------------------------------------

def find_cosine_distance(embedding1, embedding2):
    dists=np.dot(embedding1,embedding2.T).T # cosine similarity--> ranges from -1 to 1 
    dists = 1 - dists.squeeze() # cosine distance --> ranges between 0 to 2 with lower value indicating more similar
    return dists
#--------------------------------------------------------------------------------------------------------------
# not optimized
def find_min_cosine_distance(face_embedding, query_embeddings):
    dists = []
    for query_embedding in query_embeddings:
        dist=np.dot(embedding1,embedding2.T).T
        dist = np.squeeze(dist)
        dists.append(dist)

    return min(dists)

# optimized 
def is_face_in_query(face_embedding, query_embeddings, threshold=0.5):
    for query_embedding in query_embeddings:
        dist=np.dot(face_embedding,query_embedding.T).T
        dist = 1 - dist.squeeze()
        if dist<=threshold:
            print(f'Intruder found at distance: {dist}')
            return True

    return False

#--------------------------------------------------------------------------------------------------------------

# function to do the face recognition

def crop_and_find(frame, boxes, identities, face_model, query_embeddings):
    print("Running face recognition...")
    intruder_found = False
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = list(map(int, box))
        cropped_image = frame[y1:y2, x1:x2]

        faces = RetinaFace.extract_faces(cropped_image, align = True) # align and extract the image
        for face in faces:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_embedding = generate_embeddings_img(face, face_model)
            intruder_found = is_face_in_query(face_embedding, query_embeddings, threshold=0.5)
            if intruder_found:
                print(f'THE INTRUDER HAS BEEN FOUND ID: {identities[i]}')
                cv2.imwrite('found_intruder.jpg', face)
                # CODE TO RUN AFTER THE INTRUDER HAS BEEN FOUND
                return identities[i]

    return None


#--------------------------------------------------------------------------------------------------------------