import numpy as np

def convert_camel_to_underscore(name):
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def get_model_config_id(model_config):
    model_id = ''
    for config_key in model_config:
        uncamel_key = convert_camel_to_underscore(config_key)
        for word in uncamel_key.split('_'): 
            model_id += word[0]
        model_id += '_(%s)_'%(model_config[config_key],)
    return model_id

def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {0:d} samples in lengths array {1!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]
