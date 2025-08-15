import pytest
from orchestrator.test.reference_output.compare_outputs import compare_outputs

ref_dir = 'TEST_PATH/reference_output'
test_dir = 'INSTALL_PATH/'


@pytest.mark.parametrize('job_id', range(9))
def test_descriptors_local(job_id):
    """
    Tests descriptor evaluation and storage in the Local storage backend.

    In this test the KLIFFDescriptor is used to compute ACSF descriptors for a
    small selection of atomic configurations.

    :param job_id: id of the descriptor run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/descriptor/KLIFFDescriptor/01_local/'
                f'{job_id:05}/saved_config.extxyz')
    test_file = (f'{test_dir}/descriptor/computer/KLIFFDescriptor/01_local/'
                 f'{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(9))
def test_descriptors_colabfit(job_id):
    """
    Tests descriptor evaluation and storage in the Local storage backend.

    In this test the KLIFFDescriptor is used to compute ACSF descriptors for a
    small selection of atomic configurations.

    :param job_id: id of the descriptor run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/descriptor/KLIFFDescriptor/02_colabfit/'
                f'{job_id:05}/saved_config.extxyz')
    test_file = (f'{test_dir}/descriptor/computer/KLIFFDescriptor/02_colabfit/'
                 f'{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(2))
def test_descriptors_local_batched(job_id):
    """
    Tests descriptor evaluation and storage in the ColabFit storage backend.

    In this test the KLIFFDescriptor is used to compute ACSF descriptors for a
    small selection of atomic configurations. Uses batched evaluation.

    :param job_id: id of the descriptor run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/descriptor/KLIFFDescriptor/'
                f'03_local_batched/{job_id:05}/saved_config.extxyz')
    test_file = (f'{test_dir}/descriptor/computer/KLIFFDescriptor/'
                 f'03_local_batched/{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(9))
def test_descriptors_local_quests(job_id):
    """
    Tests descriptor evaluation and storage in the Local storage backend.

    In this test QUESTS is used to compute QUESTS descriptors for a
    small selection of atomic configurations.

    :param job_id: id of the descriptor run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/descriptor/QUESTSDescriptor/04_local_quests/'
                f'{job_id:05}/saved_config.extxyz')
    test_file = (
        f'{test_dir}/descriptor/computer/QUESTSDescriptor/04_local_quests/'
        f'{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(9))
def test_descriptors_colabfit_quests(job_id):
    """
    Tests descriptor evaluation and storage in the Local storage backend.

    The QUESTSDescriptor is used to compute QUESTS descriptors for a
    small selection of atomic configurations.

    :param job_id: id of the descriptor run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/descriptor/QUESTSDescriptor/05_colabfit_quests/'
                f'{job_id:05}/saved_config.extxyz')
    test_file = (
        f'{test_dir}/descriptor/computer/QUESTSDescriptor/05_colabfit_quests/'
        f'{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(2))
def test_descriptors_local_batched_quests(job_id):
    """
    Tests descriptor evaluation and storage in the ColabFit storage backend.

    The QUESTSDescriptor is used to compute QUESTS descriptors for a
    small selection of atomic configurations. Uses batched evaluation.

    :param job_id: id of the descriptor run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/descriptor/QUESTSDescriptor/'
                f'06_local_batched_quests/{job_id:05}/saved_config.extxyz')
    test_file = (f'{test_dir}/descriptor/computer/QUESTSDescriptor/'
                 f'06_local_batched_quests/{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)
