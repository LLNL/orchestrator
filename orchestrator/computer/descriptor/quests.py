from ase import Atoms
import numpy as np
from .descriptor_base import AtomCenteredDescriptor
from orchestrator.utils.data_standard import METADATA_KEY

from typing import Optional, Union, Any

from quests.descriptor import get_descriptors


class QUESTSDescriptor(AtomCenteredDescriptor):
    """
    Leverages the QUESTS library for model agnostic descriptors.
    """

    def __init__(
        self,
        num_nearest_neighbors: Optional[int] = 32,
        cutoff: Optional[float] = 5.0,
    ):
        """
        :param num_nearest_neighbors: the number of nearest neighbors
            considered in calculation. Determines the dimensionality
            of the quests descriptor: (2*num_nearest_neighbors)-1
        :type num_nearest_neighbors: int
        :param cutoff: the distance in angstroms considered in calculation
        :type cutoff: float
        """
        super().__init__()
        self.OUTPUT_KEY = 'quests_descriptor'
        self.num_nearest_neighbors = num_nearest_neighbors
        self.cutoff = cutoff

        self._metadata = {
            'num_nearest_neighbors': num_nearest_neighbors,
            'cutoff': cutoff,
        }

        self._init_args = self._metadata  # they happen to be the same for this

    def compute(
        self,
        atoms: Union[list[Atoms], Atoms],
        **kwargs,
    ) -> np.ndarray:
        """
        Computes the QUESTS descriptors for one configuration of atoms.

        :param atoms: the atomic structure to compute descriptors for
        :type atoms: ASE.Atoms object

        :returns: (N,D) array of D-dimensional QUESTS descriptors
            corresponding to the N atoms in the atomic configuration
            where D equals (2*num_nearest_neighbors)-1
        """
        if isinstance(atoms, Atoms):
            return self.compute_batch([atoms])[0]
        elif isinstance(atoms, list):
            return self.compute_batch(atoms)[0]
        else:
            raise RuntimeError(
                f"Invalid input type '{type(atoms)}' passed to .compute()")

    def compute_batch(
        self,
        list_of_atoms: list[Atoms],
        **kwargs,
    ) -> list[np.ndarray]:
        """
        Computes the QUESTS descriptors for all configurations in the list.

        :param list_of_atoms: atomic structures to compute descriptors
        :type list_of_atoms: list of ASE.Atoms objects

        :returns: list of (N, D) arrays of D-dimensional QUESTS
            descriptors corresponding to the descriptors of each
            atomic configuration of N atoms, where D equals
            (2*num_nearest_neighbors)-1
        :rtype: list
        """
        for atoms in list_of_atoms:
            # NOTE: these are being attached here because ColabFit can't do
            # nested key extraction. e.g. extracting "cut_name" from
            # atoms.info[METADATA_KEY][self.OUTPUT_KEY]['cut_name']
            for k, v in self._metadata.items():
                atoms.info[f'{self.OUTPUT_KEY}_{k}'] = v

            # to avoid overwriting if METADATA_KEY already exists
            if METADATA_KEY not in atoms.info:
                atoms.info[METADATA_KEY] = {}

            atoms.info[METADATA_KEY][self.OUTPUT_KEY] = self._metadata

        results = [
            get_descriptors([atoms],
                            k=self.num_nearest_neighbors,
                            cutoff=self.cutoff) for atoms in list_of_atoms
        ]
        return results

    def get_colabfit_property_definition(
        self,
        name: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            'property-id': 'tag:staff@noreply.colabfit.org,2024-12-09:'
            f'property/{self.OUTPUT_KEY.replace("_", "-")}',
            # kim properties don't support "_", use "-"
            "property-name": self.OUTPUT_KEY.replace('_', '-'),
            "property-title": "QUESTS descriptor",
            "property-description": "The concatenation of a list of sorted"
            "neighbor distances and average triplet bond lengths",

            "descriptors": {  # example: (N,D) arrays
                "type": "float",
                "has-unit": False,
                "extent": [":", ":"],
                "required": True,
                "description": "The per-atom descriptors. N is equal to"
                "the number of atoms and D is (2*num-nearest-neighbors)-1",
            },
            "num-nearest-neighbors": {  # example: 32
                "type": "int",
                "has-unit": False,
                "extent": [],
                "required": True,
                "description": "The number of nearest neighbors "
                "included in the calculation. Determines the dimensionality "
                "of the quests descriptor: (2*num-nearest-neighbors)-1",
            },
            "cutoff": {  # example: 5.0
                "type": "float",
                "has-unit": True,
                "extent": [],
                "required": True,
                "description": "The cutoff distance in calculation",
            },
        }

    def get_colabfit_property_map(
        self,
        name: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            'descriptors': {
                'field': self.OUTPUT_KEY + "_descriptors",
                'units': None
            },
            'num-nearest-neighbors': {
                'field': self.OUTPUT_KEY + "_num_nearest_neighbors",
                'units': None
            },
            'cutoff': {
                'field': self.OUTPUT_KEY + "_cutoff",
                'units': 'Ang'
            },
        }
