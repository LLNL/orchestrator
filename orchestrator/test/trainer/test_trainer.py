import pytest
from orchestrator.test.reference_output.compare_outputs import compare_outputs

ref_dir = 'TEST_PATH/reference_output/'
test_dir = 'INSTALL_PATH/'


@pytest.mark.parametrize('job_id', [1, 4])
def test_dnn_trainer_bp_potential(job_id):
    """
    Test the DNNTrainer and KliffBPPotential modules

    In this test a potential is generated with KliffBPPotential, and trained
    with DNNTrainer as specified in the input file. The trained model output,
    NN.params, is used for validation

    :param job_id: id of the trainer run to test, set by pytest parametrize
    :type job_id: int
    """
    if job_id == 1:
        workflow = 'trainer'
    elif job_id == 4:
        workflow = 'submit_trainer'

    ref_file = f'{ref_dir}/trainer/01_unit_test/00000/NN.params'
    test_file = (
        f'{test_dir}/trainer/{workflow}/DUNNTrainer/0{job_id}_unit_test/'
        f'00000/kim_potential/NN.params')
    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', [0])
def test_parametric_trainer_sw_potential(job_id):
    """
    Test the ParametricModelTrainer and KIMPotential modules

    In this test a potential is generated with KIMPotential (SW), and trained
    with ParametricModelTrainer as specified in the input file. The trained
    model output, kim_potential.params, is used for validation

    :param job_id: id of the trainer run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/trainer/02_unit_test/0000{job_id}/'
                f'kim_potential.params')

    test_file = (f'{test_dir}/trainer/trainer/ParametricModelTrainer/'
                 f'02_unit_test/0000{job_id}/final_model/'
                 'final_model.params')
    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', [0])
def test_new_parametric_trainer_sw_potential(job_id):
    """
    Test the ParametricModelTrainer and KIMPotential modules

    In this test a potential is generated with KIMPotential (SW), and trained
    with ParametricModelTrainer as specified in the input file. The trained
    model output, kim_potential.params, is used for validation

    :param job_id: id of the trainer run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/trainer/09_unit_test/0000{job_id}/'
                f'kim_potential.params')
    test_file = (f'{test_dir}/trainer/trainer/NewParametricModelTrainer/'
                 f'09_unit_test/0000{job_id}/kim_potential/'
                 f'kim_potential.params')
    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', [3, 5])
def test_fitsnap_trainer(job_id):

    if job_id == 3:
        workflow = 'trainer'
    elif job_id == 5:
        workflow = 'submit_trainer'

    test_file = (f'{test_dir}/trainer/{workflow}/FitSnapTrainer/0{job_id}'
                 f'_unit_test/00000/fitsnap_potential.md')
    ref_file = (f'{ref_dir}/trainer/03_unit_test/00000/'
                f'fitsnap_potential.md')
    compare_outputs(ref_file, test_file)
    test_file = (f'{test_dir}/trainer/{workflow}/FitSnapTrainer/0{job_id}'
                 f'_unit_test/00000/fitsnap_potential.snapcoeff')
    ref_file = (f'{ref_dir}/trainer/03_unit_test/00000/'
                f'fitsnap_potential.snapcoeff')
    compare_outputs(ref_file, test_file, skip_lines=1)


@pytest.mark.parametrize('job_id', [6, 7])
def test_fitsnap_weighted_trainer(job_id):

    if job_id == 6:
        workflow = 'trainer'
    elif job_id == 7:
        workflow = 'submit_trainer'

    ref_file = (f'{ref_dir}/trainer/06_unit_test/00000/'
                f'fitsnap_potential.md')
    test_file = (f'{test_dir}/trainer/{workflow}/FitSnapTrainer/0{job_id}'
                 f'_unit_test/00000/fitsnap_potential.md')
    compare_outputs(ref_file, test_file)
    ref_file = (f'{ref_dir}/trainer/06_unit_test/00000/'
                f'fitsnap_potential.snapcoeff')
    test_file = (f'{test_dir}/trainer/{workflow}/FitSnapTrainer/0{job_id}'
                 f'_unit_test/00000/fitsnap_potential.snapcoeff')
    compare_outputs(ref_file, test_file, skip_lines=1)
