The Orchestrator is an extensible and modular python framework developed as
part of a Laboratory Directed Research and Development (LDRD) Strategic
Initiaive at Lawrence Livermore National Lab. It aims to streamline the complex
workflows of building, training, and analyzing Interatomic Potentials (IAPs)
alongside the execution and analysis of large scale MD simulations made
possible by these IAPs to answer fundamental scientific questions.

Our goal is to empower researchers to more efficiently and effectively utilize
the diverse tools and approaches in this space by encoding techniques and
expertise into our software. We provide a common interface to numerous codes
alongside unique functionality to reduce human time spent on the integration
process.

# Documentation

Please check the documentation (available [here](TO BE UPDATED)) for
more information on installing dependencies, setting up the envionment to use
all supported features, and documentation on the Orchestrator as a whole,
including usage examples.

# Style Guide

This project follows the
[PEP8 style guide](https://peps.python.org/pep-0008/).
Formatting is automatically enforced using the
[yapf autoformatter](https://github.com/google/yapf) and
[flake8](https://flake8.pycqa.org/en/latest/) (including the
[pep8-naming](https://github.com/PyCQA/pep8-naming) plug-in) to
enforce style rules. Enforcement is done using
[pre-commit](https://pre-commit.com/). While the autoformatter is robust,
it is not perfect, and we highlight a handful of style conventions that
should be used when contributing to this project.

- Variables and methods are given descriptive and specific names
- 4-space indents, 79 character lines
- Class names use PascalCase
- Function and variable names use snake_case
- String formatting should use the f-strings, not `%s` or `.format()`,
whenever possible.
- Arrays, dicts, input arguments, and other collections of variables
separated by commas should generally end with a comma to trigger
`yapf` formatting. For example: `my_list = [a, b, c,]`.
- Strings should use single quotes whenever possible
- Methods include type hints
- All methods and classes include docstrings
    - Class docstrings should include a summary of the class functionality and purpose
      as well as description of any class attributes
    - `__init__()` docstrings should describe the parameters needed to instantiate the class
    - **all docstrings** should be formatted with a single line summary, blank line, descriptive
    block if more detail is needed, and parameter desctiption following the example below:
        ```
        def function_name(
            descriptive_arg1: str,
            clearly_named_arg2: Union[int,float],
            example_optional_arg3: Optional[bool] = False,
        ) -> list[Atoms]:
            """
            single line summary of what the function does (<79 chars)

            More verbose additional explanation of how the function works,
            different options for running, etc. Can be as many lines as needed
            though should be as clear and concise as possible. Note blank
            lines before and after this block.

            :param descriptive_arg1: description of what this argument means
                and how it is used. If more than one line is needed, indent
                the subsequent lines
            :type ref_fildescriptive_arg1e: str
            :param clearly_named_arg2: could list different options for the
                parameter, and what they do
            :type clearly_named_arg2: int or float
            :param example_optional_arg3: this parameter is optional, explain
                behavior with/without it. Then describe the default value if
                not provided using the following construct: |default| ``None``
            :type example_optional_arg3: bool
            :returns: describe what gets returned by the function, in this
                case a list of ASE Atoms
            :rtype: list of Atoms
            """
            ~method code~
        ```


**N.B. pre-commit must be installed for this project on a per repo
basis. You can install pre-commit with `pip install pre-commit` OR
`conda install -c conda-forge pre-commit`. In the repo, run
`pre-commit install` at the root level. This should only need to be
done once. Check to make sure `.git/hooks/pre-commit` exists.** If you follow
the setup instructions in the documentation, pre-commit will be installed, but
you will still need to run `pre-commit install` the first time in your repo.


# Contributing

If you would like to request a new feature or report a bug, please do so using
the [issues](https://github.com/LLNL/orchestrator/issues) page, which includes a formatted template and a
variety of labels to improve organization.

For contributing to the code, branches should be based of off `main` and a
[pull request](https://github.com/LLNL/orchestrator/pulls) should be opened with the template completed.
Ideally, this PR will correspond with at least one open issue, which can be
automatically tracked by naming the branch as `issue_number-branch_name`.

## Commit message codes

To help make commit messages more informative and readable, we strongly
encourage the use of commit acronyms prepending the informative message. We
are using the same codes encouraged by the
[ASE project](https://wiki.fysik.dtu.dk/ase/development/contribute.html#writing-the-commit-message),
which are copied below for ease of reference:

Standard acronyms to start the commit message with are:

**API:** an (incompatible) API change\
**BLD:** change related to building the Orchestrator\
**BUG:** bug fix\
**DEP:** deprecate something, or remove a deprecated object\
**DEV:** development tool or utility\
**DOC:** documentation\
**ENH:** enhancement\
**MAINT:** maintenance commit (refactoring, typos, etc.)\
**REV:** revert an earlier commit\
**STY:** style fix (whitespace, PEP8)\
**TST:** addition or modification of tests

An example commit message may be:

    BUG: add missing > in melting point test. Give separate npt and nph walltimes

or

    TST: added test for lammps snap oracle, differentiate from lammps kim oracle

# Release

Orchestrator is distributed under the Apache 2.0 with LLVM exception License.
All new contributions must be made under this license.

See [LICENSE](https://github.com/LLNL/orchestrator/blob/main/LICENSE.txt), and 
[NOTICE](https://github.com/LLNL/orchestrator/blob/main/NOTICE.txt) for details.

LLNL-CODE-2009974

# Contact and Questions

For questions or help where opening an issue is not a reasonable option, please
email the development team at orchestrator-help@llnl.gov.
