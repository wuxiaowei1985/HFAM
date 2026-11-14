import argparse
import re
from pathlib import Path

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RUN_PATTERN = re.compile(r"_RUN(\d+)(?:\.[^.]+)?$", re.IGNORECASE)

def extract_run_id(filename):
    match = RUN_PATTERN.search(filename)
    return int(match.group(1)) if match else None

def get_run_files(embryo_dir):
    run_files = {}
    invalid_files = []
    if not embryo_dir.exists():
        return run_files, invalid_files
    for file_path in embryo_dir.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        run_id = extract_run_id(file_path.name)
        if run_id is None:
            invalid_files.append(file_path.name)
            continue
        run_files[run_id] = file_path
    return run_files, invalid_files

def get_missing_runs(run_ids):
    if not run_ids:
        return []
    run_ids = sorted(run_ids)
    expected = set(range(run_ids[0], run_ids[-1] + 1))
    return sorted(expected - set(run_ids))

def check_embryo(dataset_root, embryo_name):
    focus_runs = {}
    invalid_files = {}
    missing_focus_dirs = []
    for focus in FOCUS_ORDER:
        focus_dir = dataset_root / ROOT_MAP[focus] / embryo_name
        if not focus_dir.exists():
            missing_focus_dirs.append(focus)
            focus_runs[focus] = {}
            invalid_files[focus] = []
            continue
        run_files, invalid = get_run_files(focus_dir)
        focus_runs[focus] = run_files
        invalid_files[focus] = invalid
    all_run_ids = sorted(set().union(*(set(runs.keys()) for runs in focus_runs.values())))
    if not all_run_ids:
        return {
            "embryo": embryo_name,
            "all_run_ids": [],
            "missing_runs": [],
            "inconsistent_runs": {},
            "invalid_files": invalid_files,
            "missing_focus_dirs": missing_focus_dirs,
            "is_valid": False,
        }
    reference_focus = "F0"
    reference_runs = set(focus_runs[reference_focus].keys())
    missing_runs = get_missing_runs(reference_runs)
    inconsistent_runs = {}
    for run_id in all_run_ids:
        missing_focus = [
            focus
            for focus in FOCUS_ORDER
            if run_id not in focus_runs[focus]
        ]
        if missing_focus:
            inconsistent_runs[run_id] = missing_focus
    is_valid = (
        not missing_runs
        and not inconsistent_runs
        and not any(invalid_files.values())
        and not missing_focus_dirs
    )
    return {
        "embryo": embryo_name,
        "all_run_ids": all_run_ids,
        "missing_runs": missing_runs,
        "inconsistent_runs": inconsistent_runs,
        "invalid_files": invalid_files,
        "missing_focus_dirs": missing_focus_dirs,
        "is_valid": is_valid,
    }

def check_dataset(dataset_root):
    dataset_root = Path(dataset_root)
    embryo_root = dataset_root / ROOT_MAP["F0"]
    if not embryo_root.exists():
        raise FileNotFoundError(f"找不到数据集目录: {embryo_root}")
    embryo_names = sorted(path.name for path in embryo_root.iterdir() if path.is_dir())
    results = []
    print("=" * 80)
    print("南特704时间序列数据检查")
    print("=" * 80)
    print(f"数据集: {dataset_root}")
    print(f"胚胎数量: {len(embryo_names)}")
    print(f"焦平面: {FOCUS_ORDER}")
    print("=" * 80)
    for embryo_name in embryo_names:
        result = check_embryo(dataset_root, embryo_name)
        results.append(result)
        if result["is_valid"]:
            print(f"[正常] {embryo_name}: {len(result['all_run_ids'])} 个RUN")
            continue
        print(f"\n[异常] {embryo_name}")
        if result["missing_runs"]:
            print(f"  F0 RUN编号存在缺失: {result['missing_runs']}")
        if result["inconsistent_runs"]:
            print("  焦平面RUN不一致:")
            for run_id, missing_focus in result["inconsistent_runs"].items():
                print(
                    f"    RUN{run_id}: "
                    f"缺失 {', '.join(missing_focus)}"
                )
        if result["missing_focus_dirs"]:
            print(
                "  缺失焦平面目录: "
                f"{', '.join(result['missing_focus_dirs'])}"
            )
        invalid = {focus: files for focus, files in result["invalid_files"].items() if files}
        if invalid:
            print("  无法解析RUN编号的文件:")
            for focus, files in invalid.items():
                print(f"    {focus}: {files}")
    return results

def print_summary(results):
    total = len(results)
    valid = sum(result["is_valid"] for result in results)
    invalid = total - valid
    missing_run_embryos = [result for result in results if result["missing_runs"]]
    inconsistent_embryos = [result for result in results if result["inconsistent_runs"]]
    invalid_file_embryos = [result for result in results if any(result["invalid_files"].values())]
    missing_dir_embryos = [result for result in results if result["missing_focus_dirs"]]
    print("\n")
    print("=" * 80)
    print("检查汇总")
    print("=" * 80)
    print(f"胚胎总数:                 {total}")
    print(f"完全正常:                 {valid}")
    print(f"存在问题:                 {invalid}")
    print(f"存在RUN编号缺失:          {len(missing_run_embryos)}")
    print(f"存在焦平面RUN不一致:      {len(inconsistent_embryos)}")
    print(f"存在无法解析文件:         {len(invalid_file_embryos)}")
    print(f"存在缺失焦平面目录:       {len(missing_dir_embryos)}")
    print("=" * 80)
    if missing_run_embryos:
        print("\n存在RUN缺失的胚胎:")
        for result in missing_run_embryos:
            print(
                f"  {result['embryo']}: "
                f"{result['missing_runs']}"
            )
    if inconsistent_embryos:
        print("\n存在焦平面不一致的胚胎:")
        for result in inconsistent_embryos:
            print(f"  {result['embryo']}:")
            for run_id, missing_focus in result["inconsistent_runs"].items():
                print(
                    f"    RUN{run_id}: "
                    f"缺失 {', '.join(missing_focus)}"
                )

def main():
    parser = argparse.ArgumentParser(description="检查南特704胚胎数据的RUN时间顺序和7焦平面完整性")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="南特704数据集根目录",
    )
    args = parser.parse_args()
    results = check_dataset(args.dataset)
    print_summary(results)

if __name__ == "__main__":
    main()