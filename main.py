"""
ERflow - Main CLI and Application Launcher
Team: Argo (Karan Aditya, Jai A Mishra) - IIT Guwahati
"""

import sys
import os
import argparse
import subprocess


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Launch FastAPI backend server with Uvicorn."""
    import uvicorn
    print(f"Starting ERflow FastAPI backend at http://{host}:{port}...")
    uvicorn.run("erflow.api.main:app", host=host, port=port, reload=True)


def run_ui(port: int = 8501):
    """Launch Streamlit frontend dashboard."""
    ui_path = os.path.join(os.path.dirname(__file__), "erflow", "ui", "app.py")
    print(f"Launching Streamlit Dashboard from {ui_path}...")
    cmd = [sys.executable, "-m", "streamlit", "run", ui_path, "--server.port", str(port)]
    # Run from the project root so .streamlit/config.toml (UI theme) is picked up
    subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


def run_tests():
    """Run full pytest verification suite."""
    print("Running ERflow Test Suite...")
    cmd = [sys.executable, "-m", "pytest", "erflow/tests", "-v"]
    subprocess.run(cmd)


def run_batch_simulation():
    """Run a batch patient assessment on synthetic ED patients."""
    from erflow.agent.orchestrator import AssessmentOrchestrator
    from erflow.scripts.generate_data import generate_synthetic_ed_data

    print("Generating 10 synthetic ED patients across pediatric, adult, and geriatric cohorts...")
    df = generate_synthetic_ed_data(n_samples=10, random_seed=42)
    orchestrator = AssessmentOrchestrator()

    print("\n" + "="*80)
    print(f"{'PATIENT ID':<12} | {'AGE/COHORT':<14} | {'ML ESI':<7} | {'FLOOR':<6} | {'FINAL':<6} | {'P(RISK)':<8} | {'CONF%':<6} | {'STATUS':<20}")
    print("="*80)

    for _, row in df.iterrows():
        p_data = row.to_dict()
        res = orchestrator.analyze_patient(p_data, wait_time_mins=float(p_data.get('current_wait_time_mins', 0.0)), use_llm=False)
        cohort_str = f"{int(p_data['age'])}y ({p_data['age_cohort'][:4]})"
        print(f"{res['patient_id']:<12} | {cohort_str:<14} | ESI {res['ml_esi_recommendation']:<3} | ESI {res['rule_acuity_floor']:<2} | ESI {res['final_esi']:<2} | {res['p_risk']*100:>5.1f}% | {res['confidence_score']:>5.1f}% | {res['agreement_state']:<20}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="ERflow Management CLI")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI backend")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--test", action="store_true", help="Run pytest verification suite")
    parser.add_argument("--simulate", action="store_true", help="Run batch simulation on synthetic ED patients")
    parser.add_argument("--port", type=int, default=8000, help="Port for server")

    args = parser.parse_args()

    if args.api:
        run_api(port=args.port)
    elif args.ui:
        run_ui()
    elif args.test:
        run_tests()
    elif args.simulate:
        run_batch_simulation()
    else:
        print("""
=============================================================
 ERFLOW - EMERGENCY DECISION SUPPORT SYSTEM
 Team: Argo (Karan Aditya, Jai A Mishra) - IIT Guwahati
=============================================================

Usage:
  python main.py --ui          # Launch Streamlit Interactive UI
  python main.py --api         # Launch FastAPI Backend Server
  python main.py --simulate    # Run terminal batch assessment simulation
  python main.py --test        # Run pytest test suite
""")


if __name__ == '__main__':
    main()

