## Setup Instructions
Download preprocessed data from Google Drive and place it in the data/{data_name} directory.  
[https://drive.google.com/drive/folders/1-vsF4rR6IG3YbG8WT3zVWRN78HJSdcFy?usp=sharing](https://drive.google.com/drive/folders/1sEMc1-q3CU4lioyvs9-SCUMPObFOFtFg?usp=sharing)

Copy the .env_sample file and rename it to .env.
If you use wandb, fill in the necessary fields like the WANDB_API_KEY.
If not, just rename the file without editing it.

Modify model options by editing a YAML file in the settings/ directory.
Choose or create a config that matches your training scenario.

Install dependencies listed in the requirements.txt file.

Run the training script using the following command:
python main.py -c {setting_file_name}.yaml
