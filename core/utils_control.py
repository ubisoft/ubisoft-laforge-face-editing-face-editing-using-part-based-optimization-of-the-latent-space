import torch
import json


def _pr_len_(m1, m2):
    d = torch.sqrt(torch.sum(torch.square(m1 - m2)))
    return d


def _pr_center_(m1, m2):
    return (m1 + m2) / 2.0

# ------------------------------------------------


class Control:
    def __init__(self, name, fn, eq_index, *args):
        self.name = name
        self.fn_ = fn
        self.eq_index = eq_index
        self.args = args

    def fn(self, mesh):
        return self.fn_(mesh, self.eq_index, *self.args)

# ------------------------------------------------


class Controller:
    @staticmethod
    def _eq_0_(m1, m2):
        return _pr_len_(m1, m2)

    @staticmethod
    def _eq_1_(m1, m2):
        return _pr_len_(m1, m2) / 2.0

    @staticmethod
    def _eq_2_(m1, m2, m3, m4):
        return (_pr_len_(m1, m2) + _pr_len_(m3, m4)) / 2.0

    @staticmethod
    def _eq_3_(m1, m2, m3, m4):
        return _pr_len_(m1, m2) / _pr_len_(m3, m4)

    @staticmethod
    def _eq_4_(m1, m2, m3, m4, m5, m6):
        return (_pr_len_(m1, m2) + _pr_len_(m3, m4)) / (2.0 * _pr_len_(m5, m6))

    @staticmethod
    def _eq_5_(m1, m2, m3, m4, m5, m6, m7, m8):
        return _pr_len_(_pr_center_(_pr_center_(m1, m2), _pr_center_(m3, m4)), _pr_center_(_pr_center_(m5, m6), _pr_center_(m7, m8)))

    @staticmethod
    def _eq_6_(m1, m2):
        return torch.abs(m1 - m2)[2]

    @staticmethod
    def _eq_7_(m1, m2, m3, m4, m5, m6, m7, m8):
        return ((_pr_len_(m1, m2) / _pr_len_(m3, m4)) + (_pr_len_(m5, m6) / _pr_len_(m7, m8))) / 2.

    _all_eqs_ = [_eq_0_, _eq_1_, _eq_2_, _eq_3_, _eq_4_, _eq_5_, _eq_6_, _eq_7_]

    # ------------------------------------------------

    def __init__(self, landmarks_path, measures_path, sc_mean, sc_std, vert_map=None, part_map=None, do_minus_one=True, inv_scale=1.):
        self.controls = []
        self.inv_scale = inv_scale

        def inv_t(mesh, idx):
            mean = sc_mean[idx]
            std = sc_std[idx]
            return (mesh[:, idx] * std + mean).squeeze()

        def no_inv(mesh, idx):
            return mesh[:, idx].squeeze()

        if sc_mean != None:
            self.fn_inv_t = inv_t
        else:
            self.fn_inv_t = no_inv

        with open(landmarks_path, 'r') as f:
            self.landmarks = json.load(f)
            self.landmark_list = []
            self.landmark_name_list = []
            self.landmark_sym = {}

            # fix landmark indices
            for landmark in self.landmarks:
                # self.landmarks[landmark] = self.landmarks[landmark]
                if do_minus_one:
                    self.landmarks[landmark] -= 1

            if 'array' in str(type(vert_map)):
                for landmark in self.landmarks:
                    self.landmarks[landmark] = vert_map[self.landmarks[landmark]]

            for idx, landmark in enumerate(self.landmarks):
                self.landmark_list.append(self.landmarks[landmark])
                self.landmark_name_list.append(landmark)

            for idx, landmark in enumerate(self.landmarks):
                if '_R' in landmark:
                    other_landmark = landmark.replace('_R', '_L')
                    self.landmark_sym[idx] = (self.landmark_list.index(self.landmarks[other_landmark]), 0)  # axis 0
                elif '_L' in landmark:
                    other_landmark = landmark.replace('_L', '_R')
                    self.landmark_sym[idx] = (self.landmark_list.index(self.landmarks[other_landmark]), 0)  # axis 0

                elif '_T' in landmark:
                    other_landmark = landmark.replace('_T', '_B')
                    self.landmark_sym[idx] = (self.landmark_list.index(self.landmarks[other_landmark]), 1)  # axis 1
                elif '_B' in landmark:
                    other_landmark = landmark.replace('_B', '_T')
                    self.landmark_sym[idx] = (self.landmark_list.index(self.landmarks[other_landmark]), 1)  # axis 1
        # part map to change only part of the latent code
        self.vert_to_part = {}
        if part_map != None:
            parts = json.load(open(part_map, 'r'))
            for idx, part in enumerate(parts):
                part_landmarks = parts[part]
                # print(f'{idx}: {part}')
                for part_landmark in part_landmarks:
                    k = self.landmark_list.index(self.landmarks[part_landmark])
                    self.vert_to_part[k] = (idx, part)

        with open(measures_path, 'r') as f:
            data = json.load(f)

            for name in data:
                eq_index = data[name]['eq']
                args = []
                args.append(self.landmarks[data[name]['m1']])
                args.append(self.landmarks[data[name]['m2']])

                if eq_index == 0 or eq_index == 1 or eq_index == 6:
                    def fn(mesh, eq_index, m1, m2):
                        return Controller._all_eqs_[eq_index].__func__(inv_scale * inv_t(mesh, m1), inv_scale * inv_t(mesh, m2))

                elif eq_index == 2 or eq_index == 3:
                    args.append(self.landmarks[data[name]['m3']])
                    args.append(self.landmarks[data[name]['m4']])

                    def fn(mesh, eq_index, m1, m2, m3, m4):
                        return Controller._all_eqs_[eq_index].__func__(inv_scale * inv_t(mesh, m1), inv_scale * inv_t(mesh, m2), inv_scale * inv_t(mesh, m3), inv_scale * inv_t(mesh, m4))

                elif eq_index == 4:
                    args.append(self.landmarks[data[name]['m3']])
                    args.append(self.landmarks[data[name]['m4']])
                    args.append(self.landmarks[data[name]['m5']])
                    args.append(self.landmarks[data[name]['m6']])

                    def fn(mesh, eq_index, m1, m2, m3, m4, m5, m6):
                        return Controller._all_eqs_[eq_index].__func__(inv_scale * inv_t(mesh, m1), inv_scale * inv_t(mesh, m2), inv_scale * inv_t(mesh, m3), inv_scale * inv_t(mesh, m4), inv_scale * inv_t(mesh, m5), inv_scale * inv_t(mesh, m6))

                elif eq_index == 5 or eq_index == 7:
                    args.append(self.landmarks[data[name]['m3']])
                    args.append(self.landmarks[data[name]['m4']])
                    args.append(self.landmarks[data[name]['m5']])
                    args.append(self.landmarks[data[name]['m6']])
                    args.append(self.landmarks[data[name]['m7']])
                    args.append(self.landmarks[data[name]['m8']])

                    def fn(mesh, eq_index, m1, m2, m3, m4, m5, m6, m7, m8):
                        return Controller._all_eqs_[eq_index].__func__(inv_scale * inv_t(mesh, m1), inv_scale * inv_t(mesh, m2), inv_scale * inv_t(mesh, m3), inv_scale * inv_t(mesh, m4), inv_scale * inv_t(mesh, m5), inv_scale * inv_t(mesh, m6), inv_scale * inv_t(mesh, m7), inv_scale * inv_t(mesh, m8))

                self.controls.append(Control(name, fn, eq_index, *args))

    def get_msr(self, mesh):
        msr = []
        for c in self.controls:
            msr.append(c.fn(mesh).item())
        return msr

    def get_landmark_pos(self, mesh, landmark_idx):
        idx = self.landmark_list[landmark_idx]
        return self.inv_scale * self.fn_inv_t(mesh, idx)

    def get_landmarks_pos(self, mesh):
        ret = []
        for landmark in self.landmarks:
            idx = self.landmarks[landmark]
            ret.append(self.inv_scale * self.fn_inv_t(mesh, idx))

        return ret

    def get_sym(self, idx):
        if idx in self.landmark_sym:
            return True, self.landmark_sym[idx]
        return False, (-1, 0)
