import logging


class Recorder:
    """
    Mixin class for logging activity in modules and functions.

    This helper class provides a standardized way to log messages from
    concrete classes. All concrete classes should inherit `Recorder` first
    in their method resolution order (MRO). To use the logging functionality,
    call `self.logger.{debug, info, warning, error}()` as needed.

    Logging configuration is set up with a default log file (`orch.log`),
    a specific format, and an INFO level. The logger is named after the
    class of the instance using it.

    This mixin class passes any arguments and keyword arguments from its
    constructor to other supers in the MRO.

    Attributes:
        logger (logging.Logger): Logger instance configured for the class.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Recorder mixin class.

        Sets up logging configuration and creates a logger instance named
        after the class of the object using it.

        :param args: Positional arguments passed to other supers in the MRO.
        :param kwargs: Keyword arguments passed to other supers in the MRO.
        """
        super().__init__(*args, **kwargs)
        logging.basicConfig(
            filename='orch.log',
            format='%(asctime)s %(name)s-%(funcName)s: %(message)s',
            level=logging.INFO,
            datefmt='%m/%d/%y %H:%M:%S',
        )
        self.logger = logging.getLogger(self.__class__.__name__)
