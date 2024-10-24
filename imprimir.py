import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Cargar el archivo CSV con los puntos faciales
csv_path = 'ruta del csv generado con el de puntos.py'
keyfacial_df = pd.read_csv(csv_path)

# Elegir una imagen aleatoria de la columna 'Image'
i = np.random.randint(0, len(keyfacial_df))
image_path = keyfacial_df['Image'][i]

# Cargar la imagen
image = cv2.imread(image_path)
# Convertir a RGB porque OpenCV carga imágenes en BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mostrar la imagen
plt.imshow(image)
plt.axis('off')  # Ocultar los ejes

# Trazar los puntos faciales
for j in range(0, 30, 2):  # Los puntos son pares (x, y)
    plt.plot(keyfacial_df.loc[i][j], keyfacial_df.loc[i][j + 1], 'rx')  # 'rx' para marcar con puntos rojos

# Mostrar la imagen con los puntos
plt.title(f"puntos faciales de la imagen")
plt.show()
