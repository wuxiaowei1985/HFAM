from pathlib import Path
from torch.utils.data import Dataset
import re
from dataset.utils.focus_loader import FocusLoader
from dataset.utils.annotation_reader import AnnotationLoader

pattern = re.compile(r"RUN(\d+)\.jpeg$")

FOCUS_ORDER = ["F-45", "F-30", "F-15", "F0", "F15", "F30", "F45"]

ROOT_MAP = {
    "F-45": "embryo_dataset_F-45",
    "F-30": "embryo_dataset_F-30",
    "F-15": "embryo_dataset_F-15",
    "F0": "embryo_dataset",
    "F15": "embryo_dataset_F15",
    "F30": "embryo_dataset_F30",
    "F45": "embryo_dataset_F45",
}

class TemporalEmbryoDataset(Dataset):
    def __init__(self, root, sequence_length=5, stride=1, drop_non_contiguous=False, transform=None, embryo_list=None):
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.drop_non_contiguous = drop_non_contiguous
        self.transform = transform
        self.loader = FocusLoader(root)
        self.annotation = AnnotationLoader(self.root / "embryo_dataset_annotations")
        self.samples = []
        self.dropped_windows = 0
        self.padded_samples = 0
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1.")
        if self.stride < 1:
            raise ValueError("stride must be >= 1.")
        embryo_root = self.root / "embryo_dataset"
        if embryo_list is None:
            embryo_dirs = [p for p in embryo_root.iterdir() if p.is_dir()]
        else:
            embryo_dirs = [embryo_root / e for e in embryo_list]
        for embryo_dir in embryo_dirs:
            embryo_name = embryo_dir.name
            run_files = {}
            for img in embryo_dir.glob("*.jpeg"):
                match = pattern.search(img.name)
                if match is None:
                    continue
                run_id = int(match.group(1))
                run_files[run_id] = img.name
            run_ids = sorted(run_files.keys())
            if not run_ids:
                continue
            if len(run_ids) < self.sequence_length:
                self._add_sample(embryo_name, run_ids, run_files)
                continue
            for start in range(0, len(run_ids) - self.sequence_length + 1, self.stride):
                window_runs = run_ids[start:start + self.sequence_length]
                if self.drop_non_contiguous and not self._is_contiguous(window_runs):
                    self.dropped_windows += 1
                    continue
                self._add_sample(embryo_name, window_runs, run_files)

    def _add_sample(self, embryo_name, window_runs, run_files):
        if not window_runs:
            return
        target_run = window_runs[-1]
        target_label = self.annotation.get_label(embryo_name, target_run)
        if target_label < 0:
            return
        valid = True
        for run_id in window_runs:
            image_name = run_files[run_id]
            for focus in FOCUS_ORDER:
                focus_path = self.root / ROOT_MAP[focus] / embryo_name / image_name
                if not focus_path.exists():
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            return
        padding_length = self.sequence_length - len(window_runs)
        if padding_length > 0:
            self.padded_samples += 1
        self.samples.append({
            "embryo": embryo_name,
            "image_names": [run_files[r] for r in window_runs],
            "runs": window_runs,
            "target_run": target_run,
            "label": target_label,
            "valid_length": len(window_runs),
            "padding_length": padding_length
        })

    @staticmethod
    def _is_contiguous(run_ids):
        return all(run_ids[i + 1] - run_ids[i] == 1 for i in range(len(run_ids) - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sequence_images = []
        for image_name in sample["image_names"]:
            images = self.loader.load_focus_images(sample["embryo"], image_name)
            sequence_images.append(images)
        if self.transform is not None:
            sequence_images = self.transform(sequence_images)
        return {
            **sample,
            "images": sequence_images
        }

if __name__ == "__main__":
    ...