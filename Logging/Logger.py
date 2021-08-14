import logging as lg
import  logging , platform
import socket

class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True

class Logger:
    def __init__(self):
        pass

    def log(self, from_file, log_type, log_message):

        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        lg.basicConfig(filename="Log_Files/main.log", level=lg.INFO,format='{} - {} - %(asctime)s - %(levelname)s -%(name)s -  %(message)s'.format(hostname,IPAddr),datefmt='%d-%b-%y %H:%M:%S')
        logger = lg.getLogger(from_file)

        if log_type == "INFO":
            logger.info( log_message + "\n")
        elif log_type == "ERROR":
            logger.error( log_message + "\n")