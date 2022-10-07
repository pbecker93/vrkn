import warnings


class ConfigDict:
    """Dictionary like class for setting, storing and accessing hyperparameters. Provides additional features:
      - setting and accessing of hyperparameters with "." (like attrdict)
      - finalize_adding(): No elements can be added after that call.
      - finalize_modifying(): Freezes the dict, no elements can be modified (or added)

      Intended use:
      1.) Implement a static method "get_default_config" for your approach, call finalize_adding() before you return
          the dict.
      2.) The user can now call the get_default_config to get the default parameters. Those can still be added but
          no new parameters can be added. This prevents (accidental) adding of parameters that are never used
      3.) Pass the configs via the __init__ to your algorithm. Call finalize_modifying() immediately in the __init__ to
          ensure the hyperparameters stay fixed from now on.
    """

    def __init__(self, **kwargs):
        """
        the configs dict will be initialized with all key value pairs in kwargs
        """
        self._adding_permitted = True
        self._modifying_permitted = True
        self._c_dict = {**kwargs}
        self._sub_confs = {}
        self._initialized = True

    def __setattr__(self, key, value):
        if "_initialized" in self.__dict__:
            if self._adding_permitted:
                self._c_dict[key] = value
            else:
                if self._modifying_permitted and key in self._c_dict.keys():
                    self._c_dict[key] = value
                elif key in self._c_dict.keys():
                    raise AssertionError("Tried modifying existing parameter after modifying finalized")
                else:
                    raise AssertionError("Tried to add parameter after adding finalized")
        else:
            self.__dict__[key] = value

    def __len__(self):
        return len(self._c_dict)

    def __getattr__(self, item):
        if "_initialized" in self.__dict__ and item in self._c_dict.keys():
            return self._c_dict[item]
        elif "_initialized" in self.__dict__ and item in self._sub_confs.keys():
            return self._sub_confs[item]
        else:
            raise AssertionError("Tried accessing non existing parameter '" + str(item) + "'")

    def get(self, key, default):
        return self._c_dict.get(key, default)

    def pop(self, key: str):
        return self._c_dict.pop(key)

    def __getitem__(self, item):
        return self._c_dict[item]

    def __setitem__(self, key, value):
        self._c_dict[key] = value

    def __str__(self):
        conf_str = "--------Config-------- \n"
        for k, v in self._c_dict.items():
            conf_str += str(k) + ": " + str(v) + "\n"
        for k, v in self._sub_confs.items():
            conf_str += "Subconfig: " + str(k) + "\n"
            sub_conf_strs = str(v).split("\n")
            for s in sub_conf_strs:
                if "---" not in s and len(s) > 1:
                    conf_str += "\t" + s + "\n"

        conf_str += "---------------------- \n "
        return conf_str

    @property
    def save_dict(self):
        raw_dict = self._c_dict.copy()
        raw_dict["_sub_confs"] = {k: v.save_dict for k, v in self._sub_confs.items()}
        return raw_dict

    @property
    def adding_permitted(self):
        return self.__dict__["_adding_permitted"]

    @property
    def modifying_permitted(self):
        return self.__dict__["_modifying_permitted"]

    def finalize_adding(self):
        self.__dict__["_adding_permitted"] = False
        for conf in self._sub_confs.values():
            conf.finalize_adding()

    def finalize_modifying(self):
        if self.__dict__["_adding_permitted"]:
            warnings.warn("ConfigDict.finalize_modifying called while adding still allowed - also deactivating adding!")
            self.__dict__["_adding_permitted"] = False
            for conf in self._sub_confs.values():
                conf.finalize_adding()
        self.__dict__["_modifying_permitted"] = False
        for conf in self._sub_confs.values():
            conf.finalize_modifying()

    def unlock_modifying(self):
        self.__dict__["_modifying_permitted"] = True
        for conf in self._sub_confs.values():
            conf.unlock_modifying()

    def unlock_adding(self):
        self.__dict__["_adding_permitted"] = True
        for conf in self._sub_confs.values():
            conf.unlock_adding()

    def keys(self):
        return self._c_dict.keys()

    def subconfig_names(self):
        return self._sub_confs.keys()

    def items(self):
        return self._c_dict.items()

    def add_subconf(self, name, sub_conf):
        self._sub_confs[name] = sub_conf

    @staticmethod
    def from_save_dict(save_dict: dict):
        conf_dict = ConfigDict(**{k: v for k, v in save_dict.items() if k != "_sub_confs"})
        for k, v in save_dict["_sub_confs"].items():
            conf_dict.add_subconf(k, ConfigDict.from_save_dict(v))
        return conf_dict

    def rec_keys(self):
        keys = list(self._c_dict.keys())
        for k, v in self._sub_confs.items():
            keys.append(k)
            keys += list(v.rec_keys())
        return keys

    def flat_rec_update(self, update_dict):
        for k, v in update_dict.items():
            if k in self._c_dict.keys():
                self._c_dict[k] = v
        for _, v in self._sub_confs.items():
            v.rec_update(update_dict)

    def rec_update(self,
                   update_dict: dict,
                   verify_all_keys_are_used: bool = True):
        used_keys = []
        for k in update_dict.keys():
            if k in self._c_dict.keys():
                self._c_dict[k] = update_dict[k]
                used_keys.append(k)
            elif self._adding_permitted and k not in self._sub_confs.keys():
                self._c_dict[k] = update_dict[k]
                used_keys.append(k)
        for k, v in self._sub_confs.items():
            if k in update_dict.keys():
                v.rec_update(update_dict[k])
                used_keys.append(k)
        if verify_all_keys_are_used:
            if set(used_keys) != set(update_dict.keys()):
                raise AssertionError("used Keys" + str(used_keys))

    def __getstate__(self) -> dict:
        state = {"c_dict": self._c_dict,
                 "adding_permitted": self._adding_permitted,
                 "modifying_permitted": self._modifying_permitted}
        for k, v in self._sub_confs.items():
            state[k] = v
        return state

    def __setstate__(self, state: dict):
        self._c_dict = state["c_dict"]
        self._adding_permitted = state["adding_permitted"]
        self._modifying_permitted = state["modifying_permitted"]
        self._sub_confs = {}
        for k, v in state.items():
            if k not in ["c_dict", "adding_permitted", "modifying_permitted"]:
                self._sub_confs[k] = ConfigDict()
                self._sub_confs[k] = v
        self._initialized = True

    def get_raw_dict(self) -> dict:
        return {k: v for k, v in self._c_dict.items()} | {k: v.get_raw_dict() for k, v in self._sub_confs.items()}

