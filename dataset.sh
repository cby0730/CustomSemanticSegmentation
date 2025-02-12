mkdir data/train
mkdir data/train/images data/train/masks
mv dataset/train/_classes.csv /home/bychen/WaveDetect/data/train/
mv dataset/train/*.jpg /home/bychen/WaveDetect/data/train/images/
mv dataset/train/*_mask.png /home/bychen/WaveDetect/data/train/masks/

mkdir data/valid
mkdir data/valid/images data/valid/masks
mv dataset/valid/_classes.csv /home/bychen/WaveDetect/data/valid/
mv dataset/valid/*.jpg /home/bychen/WaveDetect/data/valid/images/
mv dataset/valid/*_mask.png /home/bychen/WaveDetect/data/valid/masks/

rm -r dataset/train dataset/valid