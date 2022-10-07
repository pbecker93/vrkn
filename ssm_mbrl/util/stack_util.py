import torch


def stack_maybe_nested_dicts(dicts,
                             dim: int = 1):
    res = {}
    for k, v in dicts[0].items():
        if isinstance(v, torch.Tensor):
            res[k] = torch.stack([d[k] for d in dicts], dim=dim)
        else:

            if isinstance(v, tuple):
                tup_list = []
                for t_idx in range(len(v)):
                    c_list = []
                    for i in range(len(v[t_idx])):
                        if dicts[0][k][i] is None:
                            c_list.append(None)
                        else:
                            c_list.append(torch.stack([d[k][t_idx][i] for d in dicts], dim=dim))
                    tup_list.append(c_list)
                res[k] = tuple(tup_list)
            else:
                c_list = []
                for i in range(len(v)):
                    if dicts[0][k][i] is None:
                        c_list.append(None)
                    else:
                        c_list.append(torch.stack([d[k][i] for d in dicts], dim=dim))
                res[k] = c_list
    return res
