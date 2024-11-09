cd vision-sum
python -m venv env
chmod +x env/bin/activate
source env/bin/activate
pip3 install torch torchvision transformers einops mistralrs pillow 
clear
echo "Everything has been Setup Completely. Now run md.py using 'python3 md.py'."
