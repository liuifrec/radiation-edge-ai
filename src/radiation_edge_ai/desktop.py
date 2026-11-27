"""Local desktop front end over the existing offline radedge CLI (no new science).

Launch with python -m radiation_edge_ai.desktop. Tk is imported only on launch,
so ordinary command-line use and CI do not require a graphical session.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from radiation_edge_ai.desktop_runtime import Settings, build_job, run_job
from radiation_edge_ai.hardware.deployment_catalogue import REFERENCES, RUN_BACKENDS, SPOT

LABELS = {
    "Environment check (read only)": "doctor",
    "List all deployment references and candidates": "deployment-models",
    "Verify selected reference artifacts (offline)": "deployment-verify",
    "Run selected reference model": "deployment-run",
    "List pretrained models and local artifacts": "pretrained-models",
    "Scan KL720 (no model load)": "device-scan",
    "Run frozen plan (CPU or KL720)": "run-plan",
    "Run assay manifest": "run-assay",
    "Perilipin2 Track A reference (CPU, saved nuclei)": "tracka-reference",
    "Verify core record/package": "verify",
    "Export assay transaction": "export",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get(
        "RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data" if os.name == "nt"
        else str(Path.home() / "radiation-edge-ai-data")))
    args = parser.parse_args(argv)
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    repo = Path(__file__).resolve().parents[2]
    config = Path(args.data_root).expanduser().resolve() / "config" / "desktop.json"
    initial = Settings(host_python=sys.executable, data_root=args.data_root,
                       kl720_python=os.environ.get("RADEDGE_KL720_PYTHON", ""))
    config_error = None
    if config.is_file():
        try:
            saved = Settings.load(config)
            initial = Settings(sys.executable, args.data_root, saved.scientific_python,
                               saved.kl720_python, saved.kl720_port, saved.kl720_timeout_ms)
        except (OSError, ValueError, TypeError) as exc:
            config_error = str(exc)
    root = tk.Tk()
    root.title("Radiation Edge AI | Windows research launcher (experimental)")
    root.geometry("1040x760")
    frame = ttk.Frame(root, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Radiation Edge AI", font=("Segoe UI", 18, "bold")).pack(anchor="w")
    ttk.Label(frame, text=(
        "Existing NASA and DNA-fiber adapters; no Docker/CUDA needed by this launcher.\n"
        "Track A reference: recompute brown objects with saved nuclei on CPU. Perilipin2/PNKP KL720 remains disabled."
    ), wraplength=990).pack(anchor="w", pady=(4, 12))
    fields = ttk.Frame(frame)
    fields.pack(fill="x")
    fields.columnconfigure(1, weight=1)
    variables = {}
    entries = (
        ("Data / job-log root", initial.data_root, "dir"),
        ("Scientific/ONNX Python", initial.scientific_python, "file"),
        ("Existing Kneron PLUS Python", initial.kl720_python, "file"),
        ("Input manifest, transaction or Spotiflow crop", "", "file"),
        ("NEW output folder (assay, export or Spotiflow)", "", "newdir"),
    )
    def browse(key, kind):
        if kind == "file":
            value = filedialog.askopenfilename(parent=root)
        elif kind == "newdir":
            parent = filedialog.askdirectory(parent=root, title="Select parent for a NEW output folder")
            if not parent:
                return
            value = str(Path(parent) / "new_assay_result")
        else:
            value = filedialog.askdirectory(parent=root)
        if value:
            variables[key].set(value)
    for row, (key, default, kind) in enumerate(entries):
        variables[key] = tk.StringVar(value=default)
        ttk.Label(fields, text=key).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(fields, textvariable=variables[key]).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(fields, text="Browse", command=lambda k=key, t=kind: browse(k, t)).grid(row=row, column=2)
    opts = ttk.Frame(frame)
    opts.pack(fill="x", pady=8)
    ttk.Label(opts, text="USB port (blank = core auto-selection)").pack(side="left")
    port = tk.StringVar(value="" if initial.kl720_port is None else str(initial.kl720_port))
    ttk.Entry(opts, textvariable=port, width=9).pack(side="left", padx=6)
    ttk.Label(opts, text="Device timeout (ms)").pack(side="left", padx=(12, 0))
    timeout = tk.StringVar(value=str(initial.kl720_timeout_ms))
    ttk.Entry(opts, textvariable=timeout, width=10).pack(side="left", padx=6)
    action = tk.StringVar(value=next(iter(LABELS)))
    action_box = ttk.Combobox(frame, textvariable=action, values=list(LABELS), state="readonly")
    action_box.pack(fill="x", pady=5)
    selectors = ttk.Frame(frame)
    model, backend = tk.StringVar(), tk.StringVar()
    ttk.Label(selectors, text="Model").pack(side="left")
    model_box = ttk.Combobox(selectors, textvariable=model, values=list(RUN_BACKENDS), state="readonly", width=41)
    model_box.pack(side="left", padx=5)
    ttk.Label(selectors, text="Backend").pack(side="left")
    backend_box = ttk.Combobox(selectors, textvariable=backend, state="disabled", width=12)
    backend_box.pack(side="left", padx=5)

    def selection_changed(_event=None):
        action_id = LABELS[action.get()]
        enabled = action_id in ("deployment-run", "deployment-verify")
        model_box.configure(state="readonly" if enabled else "disabled")
        backend.set("")
        backend_box.configure(values=RUN_BACKENDS.get(model.get(), ()),
                              state="readonly" if action_id == "deployment-run" and model.get() else "disabled")

    selectors.pack(fill="x", pady=3)
    action_box.bind("<<ComboboxSelected>>", selection_changed)
    model_box.bind("<<ComboboxSelected>>", selection_changed)
    selection_changed()
    ttk.Label(frame, text=(
        "NASA and DNAi reference runs use their fixed panels. Leave input and output blank. "
        "Spotiflow uses one grayscale TIFF or NPY crop and a new external output folder. "
        "Track A uses its prepared bundle and scientific Python."
    ), wraplength=990).pack(anchor="w", pady=5)
    controls = ttk.Frame(frame)
    controls.pack(fill="x", pady=5)
    status = tk.StringVar(value="Ready. Preview first; device access requires confirmation.")
    ttk.Label(frame, textvariable=status, wraplength=990).pack(anchor="w", pady=5)
    log = scrolledtext.ScrolledText(frame, height=16, wrap="word", state="disabled")
    log.pack(fill="both", expand=True)
    events = queue.Queue()
    busy = False
    last_receipt = None

    def write(text):
        log.configure(state="normal")
        log.insert("end", text)
        log.see("end")
        log.configure(state="disabled")

    def settings():
        return Settings(
            sys.executable, variables[entries[0][0]].get(),
            variables[entries[1][0]].get(), variables[entries[2][0]].get(),
            int(port.get()) if port.get().strip() else None, int(timeout.get()))

    def collect():
        selected = settings()
        job = build_job(selected, repo, LABELS[action.get()],
                        variables[entries[3][0]].get(), variables[entries[4][0]].get(),
                        model=model.get(), selected_backend=backend.get())
        return selected, job

    def preview():
        try:
            _, job = collect()
            write("\nPREVIEW ONLY\n" + subprocess.list2cmdline(list(job.argv)) + "\n")
            write("Backend: {} | assay: {} | device access requested: {}\n".format(
                job.backend, job.assay_id, job.hardware_requested))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Not ready", str(exc), parent=root)

    def save():
        try:
            selected = settings()
            from radiation_edge_ai.desktop_runtime import outside_repository
            destination = Path(selected.data_root).expanduser().resolve() / "config" / "desktop.json"
            outside_repository(destination, repo)
            selected.save(destination)
            write("Settings saved: " + str(destination) + "\n")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Settings not saved", str(exc), parent=root)

    def execute():
        nonlocal busy
        if busy:
            return
        try:
            selected, job = collect()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Not ready", str(exc), parent=root)
            return
        if job.hardware_requested:
            scope = ""
            if job.action == "deployment-run":
                scope = ("Process the selected Spotiflow crop" if job.assay_id == SPOT
                         else REFERENCES[job.assay_id]["scope"]) + ".\n"
            if not messagebox.askyesno("Explicit KL720 access", scope +
                    "This action accesses the KL720 through your selected PLUS Python.\n"
                    "Close other KL720 jobs first. Continue?", parent=root):
                return
        if job.action in ("run-plan", "run-assay", "export", "tracka-reference") and not messagebox.askyesno(
                "Confirm existing workflow", "Run the selected prepared workflow?\n"
                "Track A uses saved nuclear masks, not new nuclear segmentation.\n"
                "Core artifact checks and any cached-result reuse remain unchanged.", parent=root):
            return
        busy = True
        run_button.configure(state="disabled")
        status.set("Running " + job.action + "; console output and receipts stay on the local data drive.")
        write("\n" + subprocess.list2cmdline(list(job.argv)) + "\n")
        def worker():
            try:
                receipt = run_job(job, selected, repo, lambda text: events.put(("text", text)))
                events.put(("done", receipt))
            except Exception as exc:
                events.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def poll():
        nonlocal busy, last_receipt
        while True:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break
            if kind == "text":
                write(payload)
            else:
                busy = False
                run_button.configure(state="normal")
                if kind == "done":
                    last_receipt = Path(payload)
                    record = json.loads(last_receipt.read_text(encoding="utf-8"))
                    status.set(record["status"] + "; inspect the core result and log.")
                else:
                    status.set("FAILED: " + payload)
                    write(payload + "\n")
        root.after(100, poll)

    def open_job():
        if last_receipt is None:
            messagebox.showinfo("No job yet", "Complete an action first.", parent=root)
        elif os.name == "nt":
            os.startfile(str(last_receipt.parent))
        else:
            write("Job directory: " + str(last_receipt.parent) + "\n")

    def close():
        if busy:
            messagebox.showwarning("Job still running", "Keep this window open while the current job runs.\n"
                                   "No force-stop or deletion is performed by this launcher.", parent=root)
            return
        root.destroy()

    ttk.Button(controls, text="Preview command", command=preview).pack(side="left", padx=3)
    run_button = ttk.Button(controls, text="Run selected action", command=execute)
    run_button.pack(side="left", padx=3)
    ttk.Button(controls, text="Save local settings", command=save).pack(side="left", padx=3)
    ttk.Button(controls, text="Open last job folder", command=open_job).pack(side="left", padx=3)
    if config_error:
        write("Saved settings not loaded: " + config_error + "\n")
    root.protocol("WM_DELETE_WINDOW", close)
    poll()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
