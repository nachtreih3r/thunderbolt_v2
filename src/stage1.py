from pathlib import Path
import pandas as pd

def make_csvs_from_excels(excel_folder: Path, csv_output_folder: Path) -> list[Path]:
    """
    Replicates your Stage 1:
    - Scan excel_folder for 'Daily Generation Report_*.xlsx'
    - Read only 'Steamfield' sheet with multiindex header
    - Write CSVs to csv_output_folder
    - Skip if CSV already exists
    """
    excel_folder = Path(excel_folder)
    csv_output_folder = Path(csv_output_folder)
    csv_output_folder.mkdir(parents=True, exist_ok=True)

    out_paths: list[Path] = []
    files = sorted(excel_folder.glob("Daily Generation Report_*.xlsx"))

    for xlsx in files:
        base_name = xlsx.stem
        csv_path = csv_output_folder / f"{base_name}_Steamfield.csv"
        if csv_path.exists():
            print(f"Skipping {xlsx.name} (already processed)")
            continue
        try:
            df = pd.read_excel(xlsx, sheet_name="Steamfield", header=[0, 1], skiprows=[0])
            df.to_csv(csv_path, index=False)
            print(f"Extracted Steamfield to {csv_path.name}")
            out_paths.append(csv_path)
        except Exception as e:
            print(f"Failed to process {xlsx.name}: {e}")

    return out_paths
