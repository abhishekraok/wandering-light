#!/usr/bin/env python3
"""
Launcher script for the evaluation dashboard.
"""

import os
import subprocess
import sys


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ["streamlit", "plotly", "pandas"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\n   Or if using conda:")
        print(f"   conda install {' '.join(missing_packages)}")
        return False

    print("✅ All dependencies are installed!")
    return True


def main():
    print("🚀 Evaluation Dashboard Launcher")
    print("=" * 40)

    # Check if we're in the right directory
    if not os.path.exists("wandering_light/evals/dashboard.py"):
        print("❌ Please run this script from the project root directory.")
        print(
            "   Current directory should contain 'wandering_light/evals/dashboard.py'"
        )
        return 1

    # Check dependencies
    if not check_dependencies():
        return 1

    # Check if results directory exists
    if not os.path.exists("results"):
        print("⚠️  No 'results/' directory found.")
        print("   Run some evaluations first using run_evaluation.py")
        print("   The dashboard will still start but show no data.")
        print()

    # Launch dashboard
    print("🚀 Starting Streamlit dashboard...")
    print("📊 Dashboard will open in your browser at http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Use Ctrl+C to stop the dashboard")
    print("   - Use the sidebar filters to narrow down results")
    print("   - Click on table rows to see detailed run information")
    print("\n" + "=" * 50)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "wandering_light/evals/dashboard.py",
                "--server.port",
                "8501",
                "--server.headless",
                "false",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
