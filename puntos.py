import cv2
import mediapipe as mp
import pandas as pd

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Cargar la imagen
image_path = 'ruta de la imagen que se va a convertir'
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Procesar la imagen para detectar los puntos faciales
results = face_mesh.process(rgb_image)

# Inicializar diccionario para almacenar los datos
facial_points_dict = {
    'left_eye_center_x': [None], 'left_eye_center_y': [None],
    'right_eye_center_x': [None], 'right_eye_center_y': [None],
    'left_eye_inner_corner_x': [None], 'left_eye_inner_corner_y': [None],
    'left_eye_outer_corner_x': [None], 'left_eye_outer_corner_y': [None],
    'right_eye_inner_corner_x': [None], 'right_eye_inner_corner_y': [None],
    'right_eye_outer_corner_x': [None], 'right_eye_outer_corner_y': [None],
    'left_eyebrow_inner_end_x': [None], 'left_eyebrow_inner_end_y': [None],
    'left_eyebrow_outer_end_x': [None], 'left_eyebrow_outer_end_y': [None],
    'right_eyebrow_inner_end_x': [None], 'right_eyebrow_inner_end_y': [None],
    'right_eyebrow_outer_end_x': [None], 'right_eyebrow_outer_end_y': [None],
    'nose_tip_x': [None], 'nose_tip_y': [None],
    'mouth_left_corner_x': [None], 'mouth_left_corner_y': [None],
    'mouth_right_corner_x': [None], 'mouth_right_corner_y': [None],
    'mouth_center_top_lip_x': [None], 'mouth_center_top_lip_y': [None],
    'mouth_center_bottom_lip_x': [None], 'mouth_center_bottom_lip_y': [None],
    'Image': [image_path]
}

# Verificar si se detectaron caras en la imagen
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Mapear los puntos detectados a los nombres correspondientes
        landmarks_mapping = {
            33: 'left_eye_center', 263: 'right_eye_center',
            133: 'left_eye_inner_corner', 362: 'right_eye_inner_corner',
            130: 'left_eye_outer_corner', 359: 'right_eye_outer_corner',
            55: 'left_eyebrow_inner_end', 285: 'right_eyebrow_inner_end',
            105: 'left_eyebrow_outer_end', 334: 'right_eyebrow_outer_end',
            1: 'nose_tip', 61: 'mouth_left_corner', 291: 'mouth_right_corner',
            0: 'mouth_center_top_lip', 17: 'mouth_center_bottom_lip'
        }

        for idx, landmark in enumerate(face_landmarks.landmark):
            # Obtener coordenadas escaladas
            x = landmark.x * image.shape[1]
            y = landmark.y * image.shape[0]

            # Verificar si el índice está en nuestro mapeo
            if idx in landmarks_mapping:
                key_x = f"{landmarks_mapping[idx]}_x"
                key_y = f"{landmarks_mapping[idx]}_y"
                facial_points_dict[key_x][0] = x
                facial_points_dict[key_y][0] = y

# Convertir el diccionario a un DataFrame de pandas
df_facial_points = pd.DataFrame(facial_points_dict)

# Guardar el DataFrame en un archivo CSV
output_csv_path = 'ruta en la que el csv se va a guardar'
df_facial_points.to_csv(output_csv_path, index=False)

print(f"Los puntos faciales han sido guardados en el siguiente archivo {output_csv_path}")
