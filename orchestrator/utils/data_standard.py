"""
Keys for standardizing access to common properties/fields in an ASE.Atoms
object. These are usually taken to match existing ColabFit property-definition
names.
"""
#: key for accessing energies from internal Atoms info dicts
ENERGY_KEY = 'energy_energy'
#: key for accessing forces from internal Atoms arrays
FORCES_KEY = 'atomic_forces_forces'
#: key for accessing stresses from internal Atoms info dicts
STRESS_KEY = 'cauchy_stress_stress'

#: key for accessing descriptors from internal Atoms arrays
DESCRIPTORS_KEY = 'descriptors'
#: key for accessing per-structure weights from internal Atoms info dicts
ENERGY_WEIGHT_KEY = 'structure_weight'
#: key for accessing per-atom weights from internal Atoms arrays
FORCES_WEIGHTS_KEY = 'atomic_weights'

#: key for accessing the metadata dict from internal Atoms info dicts
METADATA_KEY = '_metadata'
#: key for accessing per-atom selector boolean from internal Atoms arrays
SELECTION_MASK_KEY = 'selection_mask'

# common "utility" property maps - pass with ** prefix
# can be used to add selection_mask to the property map as:
# storage.add_property_mapping(**SELECTOR_PROPERTY_MAP)
METADATA_PROPERTY_MAP = {
    'new_property_name': METADATA_KEY,
    'new_map': {
        'metadata': {
            'field': METADATA_KEY
        }
    }
}
SELECTOR_PROPERTY_MAP = {
    'new_property_name': 'selection',
    'new_map': {
        'mask': {
            'field': SELECTION_MASK_KEY,
            'units': None
        }
    }
}

PLACEHOLDER_ARRAY_KEY = 'in_memory_array'  # used when writing args to tmp file

# For knowing if the property should be stored in .info or .arrays
PER_ATOM_PROPERTIES = ['forces', 'descriptors', 'score']
DATA_PROPERTIES = [
    f'{ENERGY_KEY}_energy',
    f'{FORCES_KEY}_forces',
    f'{STRESS_KEY}_stress',
]
