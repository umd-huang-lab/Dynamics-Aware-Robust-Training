import logging


# +
# original code
# class Logger(object):
#     """
#     Helper class for logging.
#     Arguments:
#         path (str): Path to log file.
#     """
#     def __init__(self, path):
#         self.logger = logging.getLogger()
#         self.path = path
#         self.setup_file_logger()
#         print ('Logging to file: ', self.path)
        
#     def setup_file_logger(self):
#         hdlr = logging.FileHandler(self.path, 'a') # w+: erase and write; 'a': append starting at the end
#         self.logger.addHandler(hdlr) 
#         self.logger.setLevel(logging.INFO)

#     def log(self, message):
#         print (message)
#         self.logger.info(message)
# -

class Logger(object):
    """
    Helper class for logging.
    Arguments:
        path (str): Path to log file.
    """
    def __init__(self, path):
        self.logger = logging.getLogger('__name__')
        self.path = path
        self.setup_file_logger()
        print ('Logging to file: ', self.path)
        
    def setup_file_logger(self):
        hdlr = logging.FileHandler(self.path, 'a') # w+: erase and write; 'a': append starting at the end
        hdlr.setLevel(logging.INFO)
        hdlr.setFormatter(logging.Formatter(fmt='[%(asctime)s] - %(message)s',datefmt='%Y/%m/%d %H:%M:%S'))
        self.logger.addHandler(hdlr) 
        self.logger.setLevel(logging.INFO)

    def log(self, message):
        print (message)
        self.logger.info(message)
