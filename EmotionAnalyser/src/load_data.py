from datasets import load_dataset
import re

# Cargar dataset (train/validation/test)
dataset = load_dataset("Cesar42/EmotionSpanish")

# Obtener el nombre real de la única columna (está mal formateada)
column_name = dataset["train"].column_names[0]

# Función de limpieza
def clean(example):
    raw_text = example[column_name] # A cada ejemplo se le pasa un diccionario con una clave (el nombre raro) y un valor (el string completo del ejemplo).

    # Extraer la frase
    match_frase = re.search(r"Frase:\s*(.+?)\. Clasifica", raw_text) # se busca el patrón "Frase: " seguido de cualquier texto hasta ". Clasifica"
    frase = match_frase.group(1).strip() if match_frase else ""

    # Extract the numerical label
    match_label = re.search(r"\[/INST\]\s*(\d)\s*</s>", raw_text) # se busca el patrón que contiene la etiqueta numérica
    label = int(match_label.group(1)) if match_label else -1

    return {"frase": frase, "label": label}

# Apply the cleaning function to the dataset
dataset_cleaned = dataset.map(clean)
dataset_cleaned = dataset_cleaned.remove_columns([column_name])

# Diccionario de mapeo
label2emotion = {
    0: "ira",
    1: "satisfacción",
    2: "tristeza",
    3: "culpa",
    4: "vergüenza",
    5: "miedo",
    6: "asco"
}

# Función para añadir la emoción
def add_emotion(example):
    example["emocion"] = label2emotion.get(example["label"], "desconocido")
    return example

# Aplicar el mapeo
dataset_cleaned = dataset_cleaned.map(add_emotion)

# Verify
for i in range(5):
    print(f"Ejemplo {i}:")
    print(dataset_cleaned["train"][i])




