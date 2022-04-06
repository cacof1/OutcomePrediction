import inspect


def autoload(callable_func, config_params):
    conf_param_set = set(config_params)
    req_param_set = set(inspect.signature(callable_func).parameters)
    loaded_param = {key: config_params[key] for key in conf_param_set.intersection(req_param_set)}
    return callable_func(**loaded_param)
