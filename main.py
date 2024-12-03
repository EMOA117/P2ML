import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd


class DatasetAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Loading Data ML P1")
        self.root.attributes('-zoomed', True)  # Maximiza la ventana

        # Variables
        self.separator = tk.StringVar(value=",")
        self.file_path = None
        self.dataframe = None
        self.num_neighbors = tk.IntVar(value=1)  # Número de vecinos K
        self.distance_metric = tk.StringVar(value="Manhattan")  # Métrica de distancia
        self.algorithm = tk.StringVar(value="Mínima Distancia")  # Algoritmo seleccionado
        self.input_vector = tk.StringVar(value="")  # Vector de entrada
        self.classification_attribute = ttk.Combobox(root, state="readonly") # Atributo de clasificación

        # Frame para cargar archivo
        self.file_frame = tk.Frame(self.root)
        self.file_frame.pack(pady=10)

        self.file_label = tk.Label(self.file_frame, text="Archivo:")
        self.file_label.grid(row=0, column=0, padx=5, pady=0)

        self.file_entry = tk.Entry(self.file_frame, width=40, state="readonly")
        self.file_entry.grid(row=0, column=1, padx=5, pady=0)

        self.browse_button = tk.Button(
            self.file_frame, text="Cargar Archivo", command=self.load_file
        )
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        # Frame para seleccionar separador
        self.separator_frame = tk.Frame(self.root)
        self.separator_frame.pack(pady=10)

        self.separator_label = tk.Label(self.separator_frame, text="Separador:")
        self.separator_label.grid(row=0, column=0, padx=5, pady=5)

        self.separator_entry = tk.Entry(self.separator_frame, textvariable=self.separator, width=10)
        self.separator_entry.grid(row=0, column=1, padx=5, pady=5)

        self.load_button = tk.Button(
            self.separator_frame, text="Cargar Dataset", command=self.process_file
        )
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        # Frame para mostrar información
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.info_label = tk.Label(self.info_frame, text="Información del dataset:")
        self.info_label.pack(pady=5)

        self.info_text = tk.Text(self.info_frame, height=8, state="disabled")
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Frame para selección de atributos
        self.selection_frame = tk.Frame(self.root)
        self.selection_frame.pack(pady=10)

        self.attributes_label = tk.Label(self.selection_frame, text="Seleccionar atributos:")
        self.attributes_label.grid(row=0, column=0, padx=5, pady=5)

        self.attributes_listbox = tk.Listbox(self.selection_frame, selectmode="multiple", width=40)
        self.attributes_listbox.grid(row=0, column=1, padx=5, pady=5)

        self.generate_vector_button = tk.Button(
            self.selection_frame, text="Generar Vector", command=self.generate_vector
        )
        self.generate_vector_button.grid(row=0, column=2, padx=5, pady=5)

        # Frame para ingresar parámetros de clasificación
        self.classification_frame = tk.Frame(self.root)
        self.classification_frame.pack(pady=10)

        self.attribute_label = tk.Label(self.classification_frame, text="Selecciona el atributo de clasificación:")
        self.attribute_label.grid(row=0, column=0, padx=5, pady=5)

        self.attribute_combobox = ttk.Combobox(
            self.classification_frame, textvariable=self.classification_attribute, state="readonly"
        )
        self.attribute_combobox.grid(row=0, column=1, padx=5, pady=5)

        self.vector_label = tk.Label(self.classification_frame, text="Vector de entrada (separado por comas):")
        self.vector_label.grid(row=1, column=0, padx=5, pady=5)

        self.vector_entry = tk.Entry(self.classification_frame, textvariable=self.input_vector, width=30)
        self.vector_entry.grid(row=1, column=1, padx=5, pady=5)

        self.k_label = tk.Label(self.classification_frame, text="Número de vecinos (K):")
        self.k_label.grid(row=2, column=0, padx=5, pady=5)

        self.k_entry = tk.Entry(self.classification_frame, textvariable=self.num_neighbors, width=5)
        self.k_entry.grid(row=2, column=1, padx=5, pady=5)

        self.distance_label = tk.Label(self.classification_frame, text="Distancia:")
        self.distance_label.grid(row=3, column=0, padx=5, pady=5)

        self.distance_combobox = ttk.Combobox(
            self.classification_frame,
            textvariable=self.distance_metric,
            values=["Manhattan", "Euclidiana"],
            state="readonly",
        )
        self.distance_combobox.grid(row=3, column=1, padx=5, pady=5)

        self.algorithm_label = tk.Label(self.classification_frame, text="Algoritmo:")
        self.algorithm_label.grid(row=4, column=0, padx=5, pady=5)

        self.algorithm_combobox = ttk.Combobox(
            self.classification_frame,
            textvariable=self.algorithm,
            values=["Mínima Distancia", "KNN"],
            state="readonly",
        )
        self.algorithm_combobox.grid(row=4, column=1, padx=5, pady=5)

        self.classify_button = tk.Button(
            self.classification_frame, text="Clasificar", command=self.classify
        )
        self.classify_button.grid(row=5, columnspan=2, pady=10)

    


    def load_file(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        if self.file_path:
            self.file_entry.config(state="normal")
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, self.file_path)
            self.file_entry.config(state="readonly")
        else:
            messagebox.showwarning("Error", "No se seleccionó ningún archivo.")

    def process_file(self):
        if not self.file_path:
            messagebox.showerror("Error", "Por favor, cargue un archivo primero.")
            return

        sep = self.separator.get()
        if not sep:
            messagebox.showerror("Error", "Por favor, ingrese un separador.")
            return

        try:
            self.dataframe = pd.read_csv(self.file_path, sep=sep)
            self.display_info()
            self.populate_attributes()
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar el archivo: {e}")

    def display_info(self):
        """Muestra información básica del dataset."""
        if self.dataframe is not None:
            info = []
            info.append(f"Número de atributos: {len(self.dataframe.columns)}")
            info.append(f"Número de patrones (filas): {len(self.dataframe)}\n")

            for col in self.dataframe.columns:
                if pd.api.types.is_numeric_dtype(self.dataframe[col]):
                    info.append(
                        f"Atributo '{col}' (Cuantitativo): Min={self.dataframe[col].min()}, "
                        f"Max={self.dataframe[col].max()}, Media={self.dataframe[col].mean():.2f}"
                    )
                else:
                    categories = self.dataframe[col].unique()
                    info.append(f"Atributo '{col}' (Cualitativo): Categorías={list(categories)}")

            # Mostrar en el widget de texto
            self.info_text.config(state="normal")
            self.info_text.delete("1.0", tk.END)
            self.info_text.insert(tk.END, "\n".join(info))
            self.info_text.config(state="disabled")

    def populate_attributes(self):
        """Llena el Listbox con los nombres de los atributos."""
        if self.dataframe is not None:
            self.attributes_listbox.delete(0, tk.END)
            for col in self.dataframe.columns:
                self.attributes_listbox.insert(tk.END, col)
            self.attribute_combobox['values'] = self.dataframe.columns.tolist()
            if len(self.dataframe.columns) > 0:
                self.attribute_combobox.current(len(self.dataframe.columns) - 1)

    def generate_vector(self):
        """Genera un vector con los atributos seleccionados."""
        selected_indices = self.attributes_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Advertencia", "Seleccione al menos un atributo.")
            return

        selected_columns = [self.attributes_listbox.get(i) for i in selected_indices]
        vector = self.dataframe[selected_columns].values.tolist()
        # Muestra el vector generado en un mensaje messagebox.showinfo("Vector Generado", f"Vector generado con atributos seleccionados:\n{vector}")
        return vector

    def get_labels(self):
        """Obtiene las etiquetas de clasificación."""
        selected_class = self.attribute_combobox.get()
        if not selected_class:
            messagebox.showwarning("Advertencia", "Seleccione un atributo de clasificación.")
            return

        try:
            vector = self.dataframe[selected_class].values.tolist()
            messagebox.showinfo("Vector Generado", f"Vector generado con atributo seleccionado:\n{vector}")
            return vector
        except KeyError:
            messagebox.showerror("Error", f"El atributo '{selected_class}' no está en el DataFrame.")
            return


    def euclidean_distance(vector1, vector2):
        """Calcula la distancia Euclidiana entre dos vectores."""
        return sum((x - y) ** 2 for x, y in zip(vector1, vector2)) ** 0.5

    def manhattan_distance(vector1, vector2):
        """Calcula la distancia Manhattan entre dos vectores."""
        return sum(abs(x - y) for x, y in zip(vector1, vector2))


    def knn_classification(training_data, training_labels, input_vector, k, distance_metric="euclidean"):
        """
        Clasifica un vector de entrada usando K-Nearest Neighbors.

        Args:
        - training_data: Lista de vectores de entrenamiento.
        - training_labels: Lista de etiquetas de los vectores de entrenamiento.
        - input_vector: Vector a clasificar.
        - k: Número de vecinos a considerar.
        - distance_metric: "euclidean" o "manhattan".

        Returns:
        - Clase predicha para el vector de entrada.
        """
        if len(training_data) != len(training_labels):
            raise ValueError("El tamaño de los datos de entrenamiento y las etiquetas debe ser igual.")

        # Escoger la métrica de distancia
        if distance_metric == "euclidean":
            distance_function = euclidean_distance
        elif distance_metric == "manhattan":
            distance_function = manhattan_distance
        else:
            raise ValueError("La métrica de distancia debe ser 'euclidean' o 'manhattan'.")

        # Calcular distancias
        distances = []
        for i, train_vector in enumerate(training_data):
            distance = distance_function(train_vector, input_vector)
            distances.append((distance, training_labels[i]))

        # Ordenar por distancia (los más cercanos primero)
        distances.sort(key=lambda x: x[0])

        # Obtener las etiquetas de los k vecinos más cercanos
        k_neighbors = [label for _, label in distances[:k]]

        # Contar la frecuencia de cada etiqueta
        label_count = {}
        for label in k_neighbors:
            label_count[label] = label_count.get(label, 0) + 1

        # Retornar la etiqueta con mayor frecuencia
        return max(label_count, key=label_count.get)
    
    # Nuevo método para clasificar
    def classify(self):
        input_vector = self.input_vector.get()
        training_data = self.generate_vector()
        training_labels = self.get_labels()
        k = self.num_neighbors.get()
        distance = self.distance_metric.get()
        algorithm = self.algorithm.get()

        if algorithm == "KNN":
            if not input_vector:
                messagebox.showerror("Error", "Por favor, ingrese un vector de entrada.")
                return
            if not k:
                messagebox.showerror("Error", "Por favor, ingrese un valor para K.")
                return
            return self.knn_classification(training_data, training_labels, input_vector, k, distance)
            

        # Aquí agregar lógica para la clasificación según los parámetros
        messagebox.showinfo(
            "Clasificación",
            f"Clasificando con vector={input_vector}, K={k}, Distancia={distance}, Algoritmo={algorithm}",
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetAnalyzerApp(root)
    root.geometry("800x600")
    root.mainloop()
