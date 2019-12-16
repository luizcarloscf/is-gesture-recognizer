from is_wire.core import Logger
from google.protobuf.json_format import Parse
from options_pb2 import GestureRecognizierOptions
import sys


def load_options():
    log = Logger(name='LoadingOptions')
    op_file = sys.argv[1] if len(sys.argv) > 1 else 'etc/conf/options.json'
    try:
        with open(op_file, 'r') as f:
            try:
                op = Parse(f.read(), GestureRecognizierOptions())
                log.info('GestureRecognizierOptions: \n{}', op)
                return op
            except Exception as ex:
                log.critical('Unable to load options from \'{}\'. \n{}', op_file, ex)
    except Exception as ex:
        log.critical('Unable to open file \'{}\'', op_file)