mkdir data
midir results
mkdir middle
mkdir I2S_model
mkdir S2I_model

Put data_M_wIdiom_0.csv and data_P_wIdiom_0.csv into ./data folder

Start with round 0

python data_augmentation_i2s.py --round i --mode train --model bart
python data_augmentation_i2s.py --round i --mode inference --model bart
python data_augmentation_s2i.py --round i --mode train --model bart
python data_augmentation_s2i.py --round i --mode inference --model bart