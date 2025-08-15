import pytest
from orchestrator.test.reference_output.compare_outputs import compare_outputs

ref_dir = 'TEST_PATH/reference_output/'
test_dir = 'INSTALL_PATH/'


@pytest.mark.parametrize('job_id', [0, 1, 2])
def test_espresso(job_id: int):
    """
    Test the Espresso Oracle module

    In this test the EspressoOracle is run from the provided input file. The
    processed outputs, labeled_data.xyz, are compared

    :param job_id: id of the oracle run to test, set by pytest parametrize
    """
    ref_file = (f'{ref_dir}/oracle/01_unit_test/'
                f'0000{job_id}/saved_config.extxyz')
    test_file = (f'{test_dir}/oracle/oracle/EspressoOracle/01_unit_test/'
                 f'0000{job_id}/saved_config.extxyz')
    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', [0, 1, 2])
def test_kim(job_id: int):
    """
    Test the KIM Oracle module

    In this test the KIMOracle is run from the provided input file. The
    processed outputs, labeled_data.xyz, are compared

    :param job_id: id of the oracle run to test, set by pytest parametrize
    """
    ref_file = (f'{ref_dir}/oracle/02_unit_test/'
                f'0000{job_id}/saved_config.extxyz')
    test_file = (f'{test_dir}/oracle/oracle/KIMOracle/02_unit_test/'
                 f'0000{job_id}/saved_config.extxyz')
    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', [0, 1, 2])
def test_lammps_kim(job_id: int):
    """
    Test the LAMMPS Oracle module using KIM potentials

    In this test the LAMMPSKIMOracle is run from the provided input file. The
    processed outputs, labeled_data.xyz, are compared

    :param job_id: id of the oracle run to test, set by pytest parametrize
    """
    ref_file = (f'{ref_dir}/oracle/03_unit_test/'
                f'0000{job_id}/saved_config.extxyz')
    test_file = (f'{test_dir}/oracle/oracle/LAMMPSKIMOracle/03_unit_test/'
                 f'0000{job_id}/saved_config.extxyz')
    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', [0, 1, 2])
def test_lammps_snap(job_id: int):
    """
    Test the LAMMPS Oracle module using SNAP potentials

    In this test the LAMMPSSnapOracle is run from the provided input file. The
    processed outputs, labeled_data.xyz, are compared

    :param job_id: id of the oracle run to test, set by pytest parametrize
    """
    ref_file = (f'{ref_dir}/oracle/04_unit_test/'
                f'0000{job_id}/saved_config.extxyz')
    test_file = (f'{test_dir}/oracle/oracle/LAMMPSSnapOracle/04_unit_test/'
                 f'0000{job_id}/saved_config.extxyz')
    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', [0, 1, 2])
def test_aiida_vasp(job_id: int):
    """
    Test the AiiDA Vasp Oracle module.

    In this test the AiidaVaspOracle is run from the provided input files. The
    processed outputs are compared.

    :param job_id: id of the oracle run to test, set by pytest parameterize.
    """
    ref_file = (f'{ref_dir}/oracle/05_unit_test/'
                f'0000{job_id}/saved_config.extxyz')
    test_file = (f'{test_dir}/oracle/oracle/AiidaVaspOracle/05_unit_test/'
                 f'0000{job_id}/saved_config.extxyz')
    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', [0, 1, 2])
def test_aiida_qe(job_id: int):
    """
    Test the AiiDA Quantum Espresso Oracle module.

    In this test the AiidaQEOracle is run from the provided input files. The
    processed outputs are compared.

    :param job_id: id of the oracle run to test, set by pytest parameterize.
    """
    ref_file = (f'{ref_dir}/oracle/05_unit_test/'
                f'0000{job_id}/saved_config.extxyz')
    test_file = (f'{test_dir}/oracle/oracle/AiidaEspressoOracle/06_unit_test/'
                 f'0000{job_id}/saved_config.extxyz')
    compare_outputs(ref_file, test_file)
