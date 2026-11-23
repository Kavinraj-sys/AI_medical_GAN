"""
Disease Mapping - disease_mapping.py
This file should be placed in your scripts directory
"""

# Define mapping from numerical labels to disease names
disease_label_map = {
    0: "melanoma",
    1: "basal_cell_carcinoma",
    2: "squamous_cell_carcinoma",
    3: "actinic_keratosis",
    4: "benign_keratosis",
    5: "dermatofibroma",
    6: "vascular_lesion",
    7: "nevus"
}

# Reverse mapping for convenience
label_disease_map = {v: k for k, v in disease_label_map.items()}