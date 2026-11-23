"""
disease_mapping.py - Mapping between disease codes and standardized names
Place this file in your project root directory
"""

# Mapping from disease codes/abbreviations to standardized names
disease_label_map = {
    # Format: 'code/abbreviation': 'standardized_name'
    'MEL': 'melanoma',
    'NV': 'nevus',
    'BCC': 'basal_cell_carcinoma',
    'AKIEC': 'actinic_keratosis',
    'BKL': 'benign_keratosis',
    'DF': 'dermatofibroma',
    'VASC': 'vascular_lesion',
    'SCC': 'squamous_cell_carcinoma',
    
    # Alternative names/variations
    'melanoma': 'melanoma',
    'nevus': 'nevus',
    'basal cell carcinoma': 'basal_cell_carcinoma',
    'actinic keratosis': 'actinic_keratosis',
    'benign keratosis': 'benign_keratosis',
    'dermatofibroma': 'dermatofibroma',
    'vascular lesion': 'vascular_lesion',
    'squamous cell carcinoma': 'squamous_cell_carcinoma',
    
    # Add more mappings as needed based on your dataset
}