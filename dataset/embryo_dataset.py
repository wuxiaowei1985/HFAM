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
    def __init__(self, root, sequence_length=5, stride=1, missing_frame_mode="mask", transform=None, embryo_list=None):
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.missing_frame_mode = missing_frame_mode
        self.transform = transform
        self.loader = FocusLoader(root)
        self.annotation = AnnotationLoader(self.root / "embryo_dataset_annotations")
        self.samples = []
        self.dropped_windows = 0
        self.padded_samples = 0
        self.masked_windows = 0
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1.")
        if self.stride < 1:
            raise ValueError("stride must be >= 1.")
        if self.missing_frame_mode not in {"keep", "drop", "mask"}:
            raise ValueError("missing_frame_mode must be one of {'keep', 'drop', 'mask'}.")
        embryo_root = self.root / "embryo_dataset"
        if not embryo_root.exists():
            raise FileNotFoundError(f"Embryo dataset directory does not exist: {embryo_root}")
        if embryo_list is None:
            embryo_dirs = [p for p in embryo_root.iterdir() if p.is_dir()]
        else:
            embryo_dirs = [embryo_root / e for e in embryo_list]
        for embryo_dir in embryo_dirs:
            if not embryo_dir.exists():
                continue
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
            if self.missing_frame_mode == "mask":
                self._build_mask_mode_samples(embryo_name, run_ids, run_files)
            else:
                self._build_keep_or_drop_samples(embryo_name, run_ids, run_files)

    def _build_keep_or_drop_samples(self, embryo_name, run_ids, run_files):
        if len(run_ids) < self.sequence_length:
            if self.missing_frame_mode == "drop" and not self._is_contiguous(run_ids):
                self.dropped_windows += 1
                return
            self._add_sample(embryo_name, run_ids, run_files, frame_mask=[True] * len(run_ids))
            return
        for start in range(0, len(run_ids) - self.sequence_length + 1, self.stride):
            window_runs = run_ids[start:start + self.sequence_length]
            contiguous = self._is_contiguous(window_runs)
            if self.missing_frame_mode == "drop" and not contiguous:
                self.dropped_windows += 1
                continue
            self._add_sample(embryo_name, window_runs, run_files, frame_mask=[True] * len(window_runs))

    def _build_mask_mode_samples(self, embryo_name, run_ids, run_files):
        min_run = run_ids[0]
        max_run = run_ids[-1]
        if max_run - min_run + 1 <= self.sequence_length:
            expected_runs = list(range(min_run, max_run + 1))
            frame_mask = [run_id in run_files for run_id in expected_runs]
            self._add_sample(embryo_name, expected_runs, run_files, frame_mask=frame_mask)
            return
        for start_run in range(min_run, max_run - self.sequence_length + 2, self.stride):
            window_runs = list(range(start_run, start_run + self.sequence_length))
            frame_mask = [run_id in run_files for run_id in window_runs]
            self._add_sample(embryo_name, window_runs, run_files, frame_mask=frame_mask)

    def _add_sample(self, embryo_name, window_runs, run_files, frame_mask):
        if not window_runs:
            return
        frame_mask = list(frame_mask)
        for index, (run_id, valid) in enumerate(zip(window_runs, frame_mask)):
            if not valid:
                continue
            image_name = run_files.get(run_id)
            if image_name is None:
                frame_mask[index] = False
                continue
            for focus in FOCUS_ORDER:
                focus_path = self.root / ROOT_MAP[focus] / embryo_name / image_name
                if not focus_path.exists():
                    frame_mask[index] = False
                    break
        valid_runs = [run_id for run_id, valid in zip(window_runs, frame_mask) if valid]
        if not valid_runs:
            return
        target_run = valid_runs[-1]
        target_label = self.annotation.get_label(embryo_name, target_run)
        if target_label < 0:
            return
        frame_labels = []
        for run_id, valid in zip(window_runs, frame_mask):
            if not valid:
                frame_labels.append(-100)
                continue
            frame_label = self.annotation.get_label(embryo_name, run_id)
            frame_labels.append(frame_label if frame_label >= 0 else -100)
        valid_length = sum(frame_mask)
        padding_length = self.sequence_length - len(window_runs)
        if padding_length > 0:
            self.padded_samples += 1
        if any(not value for value in frame_mask):
            self.masked_windows += 1
        self.samples.append({
            "embryo": embryo_name,
            "image_names": [run_files[r] if mask else None for r, mask in zip(window_runs, frame_mask)],
            "runs": window_runs,
            "target_run": target_run,
            "label": target_label,
            "frame_labels": frame_labels,
            "valid_length": valid_length,
            "padding_length": padding_length,
            "frame_mask": frame_mask
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
            if image_name is None:
                sequence_images.append(None)
            else:
                sequence_images.append(self.loader.load_focus_images(sample["embryo"], image_name))
        if self.transform is not None:
            sequence_images = self.transform(sequence_images)
        return {
            **sample,
            "images": sequence_images
        }

if __name__ == "__main__":
    ...