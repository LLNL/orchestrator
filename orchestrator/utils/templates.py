from os.path import isfile


class Templates:
    """
    Class for populating an external code input file with templated patterns.

    This class reads a template file, replaces specified patterns with
    corresponding values, and writes the modified content to an output file.
    If a file with the same name already exists, a numeric suffix is appended
    to avoid overwriting.

    :param template_file: Filename (including path) of the template file.
    :type template_file: str
    :param output_path: Path where the constructed file should be saved.
    :type output_path: str
    :param output_file_name: Optional name for the output file. If not
        provided, the name of the template file will be used.
    :type output_file_name: str, optional
    """

    def __init__(self, template_file, output_path, output_file_name=None):
        """
        Initialize the Templates class.

        Sets up the template file, output path, and output file name. If an
        output file name is not provided, the name of the template file will
        be used.

        :param template_file: Filename (including path) of the template file.
        :type template_file: str
        :param output_path: Path where the constructed file should be saved.
        :type output_path: str
        :param output_file_name: Optional name for the output file. If not
            provided, the name of the template file will be used.
        :type output_file_name: str, optional
        """
        self.template = template_file
        # Use the template file's name if output_file_name is not provided
        if output_file_name is None:
            output_file_name = template_file.split('/')[-1]
        self.populated_file = f'{output_path}/{output_file_name}'
        self.file_count = 1

    def replace(self, patterns, replacements):
        """
        Replace <templated> values in the template file with specific data.

        Patterns and replacements are ordered lists, where each pattern in
        the template file is replaced with the corresponding value from the
        replacements list. For example, `patterns[0]` will be replaced with
        `replacements[0]`. If a file with the same name already exists, a
        numeric suffix (`_n`, where `n` is an integer starting at 1) will be
        appended to the file name to avoid overwriting.

        :param patterns: List of patterns to replace in the template file.
        :type patterns: list of str
        :param replacements: List of values to replace the patterns with.
        :type replacements: list of str
        :returns: Name of the generated file (not including the full path).
        :rtype: str
        :raises ValueError: If the lengths of `patterns` and `replacements`
            do not match.
        """
        # Ensure patterns and replacements are of the same length
        if len(patterns) != len(replacements):
            raise ValueError("The lengths of 'patterns' and 'replacements' "
                             "must match.")

        # Determine the output file name, avoiding overwriting existing files
        if isfile(self.populated_file):
            split_name = self.populated_file.rsplit('.', 1)
            file_part1 = split_name[0]
            file_extension = split_name[1] if len(split_name) > 1 else ''
            out_name = f'{file_part1}_{self.file_count}.{file_extension}'
            self.file_count += 1
        else:
            out_name = self.populated_file

        # Process the template file and create the output file
        with open(self.template, 'rt') as template_file:
            with open(out_name, 'wt') as out_file:
                for line in template_file:
                    for pattern, replacement in zip(patterns, replacements):
                        line = line.replace(
                            f'<{pattern.upper()}>',
                            str(replacement),
                        )
                    out_file.write(line)

        # Return only the filename itself, not the full path
        return out_name.rsplit('/', 1)[-1]
