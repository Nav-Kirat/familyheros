import subprocess
import sys

def install_dependencies():
    # Ensure setuptools and wheel are up to date
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools", "wheel"])
    
    # Install requirements with extra care
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("Standard installation failed. Attempting alternative installation method.")
        # Try installing each package individually
        with open("requirements.txt", "r") as f:
            for line in f:
                package = line.strip()
                if package and not package.startswith("#"):
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    except subprocess.CalledProcessError:
                        print(f"Could not install {package}")

if __name__ == "__main__":
    install_dependencies()