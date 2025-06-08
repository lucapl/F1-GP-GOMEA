import json
import numpy as np

def logbook_encoder(obj):
    '''
    mainly for logbooks using multistats
    '''
    log = obj.chapters
    log['default'] = obj
    return log

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)