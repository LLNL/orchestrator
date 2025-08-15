"""A Collection of utility functions for FIM and FIM-matching calculations."""

from orchestrator.potential import Potential, potential_builder
from information_matching.transform import TransformBase, avail_transform


def init_potential(input_args: dict) -> Potential:
    """
    Instantiate potential object from input_args input dictionary.

    :param input_args: parameter to instantiate the potential class
    :type input_args: dict

    :returns: potential instance
    :rtype: orchestrator.potential.Potential
    """
    # Instantiate the potential
    potential = potential_builder.build(input_args['potential_type'],
                                        input_args['potential_args'])
    # Build the potential
    potential.build_potential()
    return potential


def get_column_index_to_parameter_info(parameters_optimize: dict) -> dict:
    """
    Returns a dictionary about the potential parameters and colomn or row
    index of the FIM.

    The keys of the dictionary are index of row or column of the FIM. Each
    key contain the information about the potential parameter name and its
    extent corresponding to that index.
    For the element in the FIM in row `i` and column `j`, the derivative is
    taken with parameters corresponding to those indices.

    :param parameters_optimize: Potential parameters to differentiate and
        their values.
    :type parameters_optimize: dict

    :returns: information about which potential parameter correspond to
        each row or column index of the FIM:
        idx_mapping = {0: {"parameter": <name>, "extent": <val>}, 1: {...}}
    :rtype: dict
    """
    idx_mapping = {}
    idx = 0
    for name, param_vals in parameters_optimize.items():
        # Iterate over the parameters that we specify to be tunable
        for ii, value in enumerate(param_vals):
            # Iterate over the parameter extent
            if not any(str(item).lower() == 'fix' for item in value):
                # This parameter is tunable, add the index of this
                # parameter
                idx_mapping.update({idx: {'parameter': name, 'extent': ii}})
                idx += 1
    return idx_mapping


def init_transform(transform_type: str, transform_args: dict) -> TransformBase:
    """
    Instantiate parameter transformation class.

    :param transform_type: Name of parameter transformation type, e.g.,
        "affine".
    :type transform_type: str

    :param transform_args: Arguments needed to instantiate the parameter
        transformation class.
    type transform_args: dict

    :returns: Instance of parameter transformation class.
    :rtype: TransformBase
    """
    return avail_transform[transform_type](**transform_args)


class FIMError(Exception):
    """
    Error class for FIM calculation
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
