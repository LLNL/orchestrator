import json
import numpy as np
from orchestrator.target_property.melting_point import MeltingPoint
from orchestrator.utils.setup_input import init_and_validate_module_type
from orchestrator.utils.input_output import safe_write
from random import randint


def target_property_unit_test(input_file: str) -> bool:
    """
    basic test of the target property module

    :param input_file: input file path with requisite module blocks.
        target_property is required, storage, workflow, and potential blocks
        are optional depending on the test
    :type input_file: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    target_property_inputs = test_inputs.get('target_property')
    tp_type = target_property_inputs['target_property_type']
    calculate_property_args = target_property_inputs.get(
        'calculate_property_args', {})
    target_property = init_and_validate_module_type(
        'target_property',
        test_inputs,
    )

    potential_inputs = test_inputs.get('potential')
    # TODO: See TODO below in Potential section
    randomize_kim_id = test_inputs.get('randomize_kim_id', False)

    storage = init_and_validate_module_type('storage', test_inputs)
    workflow = init_and_validate_module_type('workflow', test_inputs)

    if potential_inputs:
        # if potential is defined in the inputs, then create it
        potential = init_and_validate_module_type(
            'potential',
            potential_inputs,
            single_input_dict=True,
        )
        _ = potential.build_potential()
        # TODO: Right now, the only way to instantiate a `Potential`
        # object from pure JSON that fulfills the requirement that its
        # `save_potential_to_kimkit()` writes an KIM API-compatible
        # archive is to instantiate a `KIMPotential` from an existing
        # potential in the KIM API collection. But that potential
        # (probably) already exists in KIMKit, so if we want to test
        # writing (and cleaning) a potential from KIMKit, we should
        # change the KIM ID after building.
        # Change this after Potential module rework.
        if hasattr(potential, "kim_id") and randomize_kim_id:
            potential.kim_id = 'Test__MO_000000' + \
                ''.join([str(randint(0, 9)) for _ in range(6)]) + '_000'
    else:
        # otherwise see if the name to use is present in the input file
        potential = target_property_inputs.get('potential')
        # if neither of these is the case, then it should be in sim_params
        # nested in target_property args, and does not need to be set here

    outfile = input_file.split('/')[-1].replace('json', 'dat').replace(
        'input', 'output')
    if not target_property.args.get('estimate_error', False):
        output_data = target_property.calculate_property(
            potential=potential,
            workflow=workflow,
            storage=storage,
            **calculate_property_args,
        )
        value = output_data['property_value']
        value_std = output_data['property_std']
        value_calc_ids = output_data['calc_ids']
        value_success = output_data['success']

        if value_success is False:
            with open(f'./{outfile}', 'w') as f:
                f.write(f'Some simulations ({value_calc_ids}) failed due to '
                        f'lost atoms error, check the simulation settings and '
                        f'associated files')
        elif value_std is not None and isinstance(value, float):
            with open(f'./{outfile}', 'w') as f:
                f.write(f'Estimated output from {tp_type}: {value} with a std '
                        f'of {value_std}')
        elif value_std is None and isinstance(value, float):
            with open(f'./{outfile}', 'w') as f:
                f.write(f'Estimated output from {tp_type}: {value} with no std'
                        f'computed')
        elif isinstance(value, np.ndarray):
            np.savetxt(f'./{outfile}', value, fmt='%.7g')
        else:
            with open(f'./{outfile}', 'w') as f:
                f.write('Error parsing {tp_type} output, check orch.log file '
                        'for additional details')
    else:
        output_data = target_property.calculate_with_error(
            target_property.args['num_calculations'],
            potential=potential,
            workflow=workflow,
            storage=storage,
            **calculate_property_args,
        )
        avg_value = output_data['property_value']
        value_std = output_data['property_std']
        # value_calc_ids = output_data['calc_ids']
        if value_std is not None and isinstance(value, float):
            with open(f'./{outfile}', 'w') as f:
                f.write(f'Estimated average output from {tp_type} with error '
                        f' calculation: {avg_value} with a std of {value_std}')
        else:
            with open(f'./{outfile}', 'w') as f:
                f.write('Error parsing {tp_type} output, check orch.log file '
                        'for additional details')

    return True


def sample_config_unit_test() -> bool:
    """
    basic test of the sampling configs from melting point calculations

    The test takes an existing trajectory (the location of the trajectory
    can be provided via ``in_paths``), sample configurations from it based on
    beginning frame (1), ending frame (5) and writing frequency (2), and save
    a new xyx trajectory. For example, the test below will write frames
    1, 3 and 5 from dump.lammpstrj and save it to sampled_configs_npt.xyz.
    The locations of the trajectories can be also retrieved by providing
    ``calc_ids`` and ``workflow`` parameters

    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """

    # beginning frame, ending frame, save every nth step
    # trajectory name, calc_ids=None, workflow=None, path of the trajectory
    atoms = MeltingPoint.sample_configs(
        1,
        5,
        2,
        "dump.lammpstrj",
        in_paths=['../sample_npt_traj'],
    )
    safe_write('sampled_configs_npt.xyz', atoms[0])

    return True
