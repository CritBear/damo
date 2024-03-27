

def dict2class(res):
    class result_meta(object):
        pass

    res_class = result_meta()
    for k, v in res.items():
        res_class.__setattr__(k, v)

    return res_class