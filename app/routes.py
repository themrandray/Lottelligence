import pandas as pd
# Flask komponentes:
# - Blueprint: ļauj sadalīt maršrutus pa moduļiem
# - render_template: ielādē HTML veidnes
# - request: nolasa formu datus un augšupielādētos failus
from flask import Blueprint, render_template, request, jsonify

# Pathlib nodrošina ērtu un drošu darbu ar failu ceļiem (platformneatkarīgi)
from pathlib import Path

# datetime tiek izmantots laika zīmoga (timestamp) izveidei saglabātajiem failiem
from datetime import datetime

# Projekta servisi datu apstrādei:
# - read_table: nolasa CSV/XLSX failu pandas DataFrame formātā
# - normalize_any: normalizē datus un pārbauda loterijas tipu
from .services.dataset import read_table, normalize_any, get_top_numbers, get_top_combinations

# Eksperimentu izpilde (modeļi, prognozes utt.)
from .services.experiment import run_experiment, summarize_experiment_results

main_bp = Blueprint("main", __name__)

@main_bp.route("/", methods=["GET"])
def index():
    # Atgriež sākuma lapu ar noklusējuma iestatījumiem
    return render_template(
        "index.html",
        error=None,
        results=None,
        last_uploaded_file=None,
        form_state={
            "lottery": "viking",
            "file_format": "raw",
            "split_ratio": "70_30",
        },
        status="idle",
        best_summary=None,
    )

@main_bp.route("/run", methods=["POST"])
def run():
    # Apstrādā augšupielādēto failu un palaiž eksperimentu
    error = None
    results = None
    last_uploaded_file = None
    best_summary = None
    # Nolasa formā ievadītos parametrus
    lottery = request.form.get("lottery", "viking")
    file_format = request.form.get("file_format", "raw")
    split_ratio = request.form.get("split_ratio", "70_30")

    form_state = {
        "lottery": lottery,
        "file_format": file_format,
        "split_ratio": split_ratio,
    }

    upload_dir = Path(main_bp.root_path).resolve().parent / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = Path(main_bp.root_path).resolve().parent / "outputs"
    normalized_latest_path = outputs_dir / "normalized_latest.csv"

    file = request.files.get("dataset")
    df_norm = None

    try:
        if file and file.filename:
            safe_name = file.filename.replace(" ", "_")
            last_uploaded_file = safe_name
            saved_path = upload_dir / safe_name
            file.save(saved_path)

            df_raw = read_table(saved_path)
            df_norm = normalize_any(df_raw, lottery=lottery, file_format=file_format)

        elif normalized_latest_path.exists():
            df_norm = read_table(normalized_latest_path)

        else:
            error = "Lūdzu augšupielādējiet datu failu"
            return render_template(
                "index.html",
                error=error,
                results=None,
                last_uploaded_file=last_uploaded_file,
                form_state=form_state,
                status="idle",
            )
        
        # Palaiž eksperimentu
        results = run_experiment(
            df_norm,
            lottery,
            split_ratio=split_ratio
        )
        best_summary = summarize_experiment_results(results)

        # Saglabā rezultātus
        _save_outputs(df_norm, results, lottery)

        status = "done"

    except Exception as exc:
        error = str(exc)
        status = "idle"

    return render_template(
        "index.html",
        error=error,
        results=results,
        last_uploaded_file=last_uploaded_file,
        form_state=form_state,
        status=status,
        best_summary=best_summary if results else None,
    )

@main_bp.route("/top-numbers-api", methods=["POST"])
def top_numbers_api():
    outputs_dir = Path(main_bp.root_path).resolve().parent / "outputs"
    normalized_latest_path = outputs_dir / "normalized_latest.csv"

    if not normalized_latest_path.exists():
        return jsonify({
            "ok": False,
            "error": "Nav saglabātu normalizētu datu. Vispirms palaidiet eksperimentu."
        })

    try:
        df_norm = read_table(normalized_latest_path)
        top_k = request.form.get("top_k", "5")

        try:
            top_k = int(top_k)
            if top_k < 1:
                top_k = 1
            elif top_k > 10:
                top_k = 10
        except ValueError:
            top_k = 5

        top_numbers = get_top_numbers(df_norm, k=top_k)

        has_n6 = "n6" in df_norm.columns and df_norm["n6"].notna().any()
        analysis_label = "Viking Lotto" if has_n6 else "Eurojackpot"

        return jsonify({
            "ok": True,
            "analysis_label": analysis_label,
            "top_numbers": [
                {"number": number, "freq": freq}
                for number, freq in top_numbers
            ]
        })

    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc)
        })

@main_bp.route("/top-combinations-api", methods=["POST"])
def top_combinations_api():
    outputs_dir = Path(main_bp.root_path).resolve().parent / "outputs"
    normalized_latest_path = outputs_dir / "normalized_latest.csv"

    if not normalized_latest_path.exists():
        return jsonify({
            "ok": False,
            "error": "Nav saglabātu normalizētu datu. Vispirms palaidiet eksperimentu."
        })

    try:
        df_norm = read_table(normalized_latest_path)

        comb_size = request.form.get("comb_size", "2")
        top_k = request.form.get("top_k", "5")

        try:
            comb_size = int(comb_size)
            if comb_size < 2:
                comb_size = 2
            elif comb_size > 4:
                comb_size = 4
        except ValueError:
            comb_size = 2

        try:
            top_k = int(top_k)
            if top_k < 1:
                top_k = 1
            elif top_k > 10:
                top_k = 10
        except ValueError:
            top_k = 5

        top_combinations = get_top_combinations(df_norm, comb_size=comb_size, top_k=top_k)

        has_n6 = "n6" in df_norm.columns and df_norm["n6"].notna().any()
        analysis_label = "Viking Lotto" if has_n6 else "Eurojackpot"

        return jsonify({
            "ok": True,
            "analysis_label": analysis_label,
            "top_combinations": [
                {"combo": list(combo), "freq": freq}
                for combo, freq in top_combinations
            ]
        })

    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc)
        })

def _timestamp():
    # Izveido laika zīmogu vēstures ierakstiem (datums + laiks milisekundēs)
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(now.microsecond / 1000):03d}"

def _save_outputs(df_norm, results, lottery):
    # Saglabā tikai trīs failus:
    # - normalized_latest.csv (pēdējais normalizētais datasets)
    # - results_latest.csv (pēdējie eksperimenta rezultāti)
    # - results_history.csv (visu skrējienu vēsture)

    from flask import current_app

    outputs_dir = current_app.config["OUTPUTS_DIR"]

    # 1) Saglabā pēdējo normalizēto datasetu
    df_norm.to_csv(outputs_dir / "normalized_latest.csv", index=False)

    # 2) Saglabā pēdējos rezultātus
    df_res = pd.DataFrame(results)
    df_res.to_csv(outputs_dir / "results_latest.csv", index=False)

    # 3) Pievieno rezultātus vēsturei
    history_path = outputs_dir / "results_history.csv"
    df_res_with_meta = df_res.copy()
    df_res_with_meta["timestamp"] = _timestamp()
    df_res_with_meta["lottery"] = lottery

    if history_path.exists():
        df_old = pd.read_csv(history_path)
        df_all = pd.concat([df_old, df_res_with_meta], ignore_index=True)
        df_all.to_csv(history_path, index=False)
    else:
        df_res_with_meta.to_csv(history_path, index=False)