"""
Build script for t-SNE Explorer using Nuitka
Bundles the application into a standalone executable
"""

import os
import sys
import subprocess
from pathlib import Path

def build_with_nuitka():
    """Build the application using Nuitka"""

    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    app_file = current_dir / "app.py"
    web_folder = current_dir / "web"
    icon_file = current_dir / "t-sne-transparent.ico"

    # Check if app.py exists
    if not app_file.exists():
        print(f"Error: app.py not found at {app_file}")
        sys.exit(1)

    # Check if web folder exists
    if not web_folder.exists():
        print(f"Error: web folder not found at {web_folder}")
        sys.exit(1)

    print("=" * 60)
    print("Building t-SNE Explorer with Nuitka")
    print("=" * 60)
    print(f"App file: {app_file}")
    print(f"Web folder: {web_folder}")
    print(f"Icon file: {icon_file if icon_file.exists() else 'Not found'}")
    print()

    # Nuitka command
    nuitka_cmd = [
        sys.executable,
        "-m", "nuitka",

        # Main file to compile
        str(app_file),

        # Onefile mode - single executable with all dependencies
        "--onefile",
        "--standalone",

        # Follow all imports
        "--follow-imports",

        # Enable plugins for specific packages
        "--enable-plugin=numpy",

        # Include data files
        f"--include-data-dir={web_folder}=web",

        # Output directory
        "--output-dir=build",

        # Windows-specific options
        "--windows-console-mode=disable" if sys.platform == "win32" else "",
        f"--windows-icon-from-ico={icon_file}" if sys.platform == "win32" and icon_file.exists() else "",

        # Optimization
        "--lto=no",

        # Show progress
        "--show-progress",

        # Show memory usage
        "--show-memory",

        # Product information
        "--product-name=t-SNE Explorer",
        "--file-description=Transparent t-SNE Visualization Tool",
        "--company-name=t-SNE Explorer",
        "--product-version=6.0",
        "--file-version=6.0.0.0",

        # Exclude problematic webview platforms
        "--nofollow-import-to=webview.platforms.android",
        "--nofollow-import-to=webview.platforms.cocoa",
        "--nofollow-import-to=webview.platforms.gtk",
        "--nofollow-import-to=webview.platforms.qt",

        # Include only Windows platform for webview
        "--include-module=webview.platforms.winforms" if sys.platform == "win32" else "",

        # Include hidden imports
        "--include-module=numpy",
        "--include-module=pandas",
        "--include-module=PIL",
        "--include-module=sklearn",
        "--include-module=sklearn.manifold",
        "--include-module=sklearn.datasets",
    ]

    # Remove empty strings from command
    nuitka_cmd = [arg for arg in nuitka_cmd if arg]

    print("Running Nuitka with the following command:")
    print(" ".join(nuitka_cmd))
    print()
    print("This may take several minutes...")
    print()

    try:
        # Run Nuitka
        result = subprocess.run(nuitka_cmd, check=True)

        print()
        print("=" * 60)
        print("Build successful!")
        print("=" * 60)
        print()
        print("The executable can be found in:")
        print(f"  {current_dir / 'build' / 'app.exe'}")
        print()
        print("To run the application:")
        print("  Double-click app.exe or run it from command line")
        print()

        return 0

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("Build failed!")
        print("=" * 60)
        print(f"Error code: {e.returncode}")
        print()
        print("Common issues:")
        print("  1. Nuitka not installed: pip install nuitka")
        print("  2. Missing C compiler (Windows: install Visual Studio)")
        print("  3. Insufficient memory (close other applications)")
        print()
        return 1

    except Exception as e:
        print()
        print("=" * 60)
        print("Unexpected error!")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        return 1


if __name__ == "__main__":
    # Check if nuitka is installed
    try:
        import nuitka
        print("Nuitka is installed")
        print()
    except ImportError:
        print("Error: Nuitka is not installed")
        print()
        print("Install it with:")
        print("  pip install nuitka")
        print()
        print("On Windows, you also need a C compiler:")
        print("  - Visual Studio 2017 or newer (recommended)")
        print("  - Or MinGW64")
        print()
        sys.exit(1)

    # Run the build
    sys.exit(build_with_nuitka())
