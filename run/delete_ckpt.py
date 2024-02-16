
import sys
sys.path.append("../../Self-Supervised-Learning/")
import os
from run.utils.plCLI import MyLightningCLI, _get_ckpt,_changeDefaultRootDir
import glob

def main():
    # Retrieve arguments
    root_dir = sys.argv[1]
    mode = sys.argv[2]
    version_num = int(sys.argv[3]) # pass in -1 to figure out newest version


    ckpt_path = _get_ckpt (root_dir, [mode, version_num] , '')
    
    try:
        os.remove(ckpt_path)
        print(f"File '{ckpt_path}' deleted successfully.")
    except OSError as e:
        print(f"Error deleting file '{ckpt_path}': {e}")
    

if __name__ == "__main__":
    main()