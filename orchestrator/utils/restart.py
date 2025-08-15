import json
from .recorder import Recorder


class Restart(Recorder):
    """
    Utility class for managing reading and writing checkpoint files.

    The `restarter` instance can be imported for access to the read and write
    checkpoint methods.
    """

    def __init__(self):
        """
        Initialize the Restart utility class.

        Provides methods to read and write checkpoint files. Tracks whether
        a checkpoint file is missing to avoid duplicate log messages.
        """
        super().__init__()
        self.file_missing_known = False

    def write_checkpoint_file(self, checkpoint_file, checkpoint_dict):
        """
        Add (or update) a checkpoint dictionary in the checkpoint file.

        If the checkpoint file already exists, the new dictionary will be
        merged with the existing content. If the module's section already
        exists, it will be overwritten with the latest data.

        :param checkpoint_file: Path and filename (should be JSON) to read
            and write to.
        :type checkpoint_file: str
        :param checkpoint_dict: Checkpoint information to add or update in the
            file. The dictionary should be nested, with a single root key
            associated with the module, and the value being a dictionary of
            quantities to checkpoint.
        :type checkpoint_dict: dict
        :raises ValueError: If the checkpoint dictionary does not have a
            single root key.
        """
        checkpoint_name = list(checkpoint_dict.keys())
        if len(checkpoint_name) != 1:
            checkpoint_name = 'ambiguous, not a single key in the dict'
        else:
            checkpoint_name = checkpoint_name[0]
        self.logger.info(f'Checkpointing: {checkpoint_name}')
        try:
            # If checkpoint file exists, merge this dict with existing content
            with open(checkpoint_file, 'r') as fin:
                previous_checkpoint = json.load(fin)
            # Merge the two dictionaries, overwriting the module's section
            output_dict = previous_checkpoint | checkpoint_dict
        except FileNotFoundError:
            # Otherwise, this dict is the only thing to save
            output_dict = checkpoint_dict
        # Save the full checkpoint content
        with open(checkpoint_file, 'w') as fout:
            json.dump(output_dict, fout, indent=4)
        self.file_missing_known = False

    def read_checkpoint_file(self, checkpoint_file, module_key):
        """
        Read a checkpoint file and extract a specific module dictionary.

        If the checkpoint file or module key does not exist, an empty
        dictionary will be returned.

        :param checkpoint_file: JSON filename (including full path) to read
            from.
        :type checkpoint_file: str
        :param module_key: Module key for the specific checkpoint dictionary
            to return.
        :type module_key: str
        :returns: Module dictionary associated with the module_key.
        :rtype: dict
        """
        try:
            with open(checkpoint_file, 'r') as fin:
                restart_dict = json.load(fin)
            try:
                module_restart = restart_dict[module_key]
            except KeyError:
                self.logger.info(f'Checkpoint file exists, but no section for '
                                 f'{module_key}')
                module_restart = {}
        except FileNotFoundError:
            module_restart = {}
            # Log the message only if a write has occurred since the last read
            if not self.file_missing_known:
                self.logger.info(f'{checkpoint_file} does not exist, starting '
                                 'from scratch')
            self.file_missing_known = True

        return module_restart


#: Restart instance which can be imported for use in other modules.
restarter = Restart()
