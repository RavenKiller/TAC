import logging


class MyLogger(logging.Logger):
    def __init__(
        self,
        name,
        level,
        filename=None,
        filemode="a",
        stream=None,
        format_str=None,
        dateformat=None,
        style="{",
    ):
        super().__init__(name, level)
        if filename is not None:
            handler = logging.FileHandler(filename, filemode)  # type:ignore
        else:
            handler = logging.StreamHandler(stream)  # type:ignore
        self._formatter = logging.Formatter(format_str, dateformat, style)
        handler.setFormatter(self._formatter)
        super().addHandler(handler)

    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)


logger = MyLogger(
    name="mylogger",
    level=logging.DEBUG,
    format_str="{asctime:15s} [{levelname:s}]\t{message}",
)
