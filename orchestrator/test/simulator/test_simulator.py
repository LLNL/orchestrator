import pytest
from orchestrator.test.reference_output.compare_outputs import compare_outputs

ref_dir = 'TEST_PATH/reference_output'
test_dir = 'INSTALL_PATH'


@pytest.mark.parametrize('job_id', [0])
def test_lammps(job_id):
    """
    Test the LAMMPS simulator module

    In this test the LAMMPSSimulator is run from the provided input file. A set
    of the processed outputs are saved to parsed_excerpts.dat and are the basis
    for comparison in this test

    :param job_id: id of the simulator run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/simulator/01_unit_test/'
                f'0000{job_id}/parsed_excerpts.dat')
    test_file = (f'{test_dir}/simulator/simulator/LAMMPSSimulator/'
                 f'01_unit_test/0000{job_id}/parsed_excerpts.dat')
    compare_outputs(ref_file, test_file)
