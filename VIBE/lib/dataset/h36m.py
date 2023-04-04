from VIBE.lib.dataset import Dataset3D
from VIBE.lib.core.config import H36M_DIR
import joblib
import os.path as osp
from collections import defaultdict
from VIBE.lib.core.config import VIBE_DB_DIR

class H36M(Dataset3D):
    def __init__(self, set, seqlen, db_name = 'h36m', overlap=0.75, debug=False):
        # during testing we don't need data augmentation
        # but we can use it as an ensemble
        super(H36M, self).__init__(
            set=set,
            folder=H36M_DIR,
            seqlen=seqlen,
            overlap=0,
            dataset_name=db_name,
            debug=debug,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')

    def load_db(self):
        split = self.set
        if split == 'train':
            user_list = [1, 5, 6, 7, 8]
        elif split == 'val':
            user_list = [9, 11]

        seq_db_list = []
        for user_i in user_list:
            print(f"Loading Subject S{user_i}" )
            db_subset = joblib.load(osp.join(VIBE_DB_DIR, f'h36m_{user_i}_db.pt'))
            seq_db_list.append(db_subset)

        dataset = defaultdict(list)
        for seq_db in seq_db_list:
            for k, v in seq_db.items():
                dataset[k] += list(v)
        return dataset

        print(f'Loaded {self.dataset_name}')
        return db
